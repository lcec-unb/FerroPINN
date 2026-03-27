"""
main.py
=======
Ponto de entrada principal do solver PINN para cavidade cisalhante.
Suporta caso hidrodinâmico puro e caso magnético (ferrofluido).

USO
---
  Varredura de parâmetros:
      python main.py sweep_config.json

  Simulação única:
      python main.py params.json --single

O JSON deve estar no mesmo diretório que este arquivo.

ESTRUTURA DO PROJETO
--------------------
  main.py                    ← este arquivo
  sweep_config.json          ← configuração (mesmo nível)

  config/
    parameters.py            ← leitura do JSON, seed, device

  core/
    network.py               ← build_net()
    sampling.py              ← generate_interior(), generate_bc()
    loss.py                  ← compute_loss() (N-S + magnético)
    weights.py               ← update_weights() (grad-norm adaptativo)

  output/
    folders.py               ← criação de pastas, logs, info_execucao.json
    export.py                ← export_results(), plot_losses()

  sweep/
    orchestrator.py          ← carregar_config(), run_sweep()

PARÂMETROS DO JSON
------------------
  Obrigatórios:
    Re, N_int, N_bc, epochs, layers, neurons

  Opcionais (com padrão):
    activation, optimizer, lr, use_lhs, switch_opt, use_norm,
    use_compile, use_amp, resample_every, log_flush_every, seed,
    w_f, w_u_top, w_u_rest, w_v,
    adaptive_weights, grad_update_every

  Caso magnético (exige magnetic_case=true):
    magnetic_case, H0, chi, D (opcional, padrão 1.0)
"""

import json
import os
import sys
import time

import numpy as np
import torch

# --- Módulos do projeto ---
from config.parameters import load_params, setup_seed, get_device
from core.network      import build_net
from core.sampling     import generate_interior, generate_bc
from core.loss         import compute_loss
from core.weights      import update_weights
from output.folders    import (make_output_dir, save_params_json,
                                open_log, save_execution_info)
from output.export     import export_results, plot_losses
from sweep.orchestrator import run_sweep


# ==============================================================================
# SIMULAÇÃO ÚNICA
# ==============================================================================
def run_simulation(params: dict) -> dict:
    """
    Executa uma simulação PINN completa a partir de um dicionário de parâmetros.

    Parâmetros
    ----------
    params : dicionário bruto vindo do JSON (ou montado pelo sweep)

    Retorna
    -------
    dict com métricas finais: output_dir, loss_final, u_avg_final,
    tempo_total_s e (se magnético) Mn.
    """

    # -------------------------------------------------------------------------
    # 1. Leitura e validação de todos os parâmetros
    # -------------------------------------------------------------------------
    p = load_params(params)

    setup_seed(p["seed"])
    device, use_amp = get_device(p["use_amp"])

    print(f"  Dispositivo: {device} | AMP: {use_amp} | seed: {p['seed']}")
    if p["magnetic_case"]:
        print(f"  H0={p['H0']} | chi={p['chi']} | Mn={p['Mn']:.4e}")

    # -------------------------------------------------------------------------
    # 2. Pasta de saída e log
    # -------------------------------------------------------------------------
    nome_caso = make_output_dir(p, output_dir_override=params.get("output_dir"))
    save_params_json(nome_caso, p)

    log_file = open_log(
        nome_caso,
        adaptive_weights  = p["adaptive_weights"],
        grad_update_every = p["grad_update_every"],
        opt_name          = p["opt_name"],
        opt_lr            = p["opt_lr"],
        switch_opt        = p["switch_opt"],
        magnetic_case     = p["magnetic_case"],
    )

    # -------------------------------------------------------------------------
    # 3. Modelo
    # -------------------------------------------------------------------------
    model = build_net(p["layers"], p["neurons"], p["act_class"], p["use_norm"])
    model = model.to(device)

    if p["use_compile"]:
        try:
            model = torch.compile(model)
            print("  torch.compile ativado")
        except Exception:
            print("  torch.compile nao disponivel (requer PyTorch >= 2.0)")

    # -------------------------------------------------------------------------
    # 4. Dados de colocação
    # -------------------------------------------------------------------------
    x_f              = generate_interior(p["N_int"], p["use_lhs"], device)
    x_bc, u_bc, v_bc = generate_bc(p["N_bc"], device)

    # Pontos da tampa pré-alocados fora do loop (evita realocação)
    pts_tampa = torch.cat([
        torch.linspace(0, 1, p["N_bc"], device=device).view(-1, 1),
        torch.ones(p["N_bc"], 1, device=device),
    ], dim=1)

    # -------------------------------------------------------------------------
    # 5. Otimizador e AMP scaler
    # -------------------------------------------------------------------------
    optimizer = p["opt_class"](model.parameters(), lr=p["opt_lr"])
    scaler    = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"  Otimizador: {p['opt_name']} | lr: {p['opt_lr']}")

    # -------------------------------------------------------------------------
    # 6. Estado mutável durante o treinamento
    # -------------------------------------------------------------------------
    weights         = p["weights"]      # dict {w_f, w_u_top, w_u_rest, w_v}
    loss_log        = []
    log_buffer      = []
    tempo_inicio    = time.time()

    # Atalhos de legibilidade
    nu             = p["nu"]
    magnetic_case  = p["magnetic_case"]
    H0             = p["H0"]
    chi            = p["chi"]
    D              = p["D"]
    Mn             = p["Mn"]
    adaptive       = p["adaptive_weights"]
    grad_every     = p["grad_update_every"]
    resample_every = p["resample_every"]
    log_flush      = p["log_flush_every"]
    switch_opt     = p["switch_opt"]
    epochs         = p["epochs"]

    # -------------------------------------------------------------------------
    # 7. Loop de treinamento
    # -------------------------------------------------------------------------
    for epoch in range(epochs):
        model.train()

        # --- Passo de otimização ---
        if isinstance(optimizer, torch.optim.LBFGS):
            # LBFGS requer closure
            def closure():
                optimizer.zero_grad()
                _, _, _, _, loss_c = compute_loss(
                    model, x_f, x_bc, u_bc, v_bc, nu, weights,
                    magnetic_case=magnetic_case, H0=H0, chi=chi, D=D,
                )
                loss_c.backward()
                return loss_c
            optimizer.step(closure)

            # Recalcula para logging (grafo foi liberado pelo closure)
            loss_f, loss_u_top, loss_u_rest, loss_v, loss_total = compute_loss(
                model,
                x_f.detach().requires_grad_(True),
                x_bc, u_bc, v_bc, nu, weights,
                magnetic_case=magnetic_case, H0=H0, chi=chi, D=D,
            )
        else:
            optimizer.zero_grad()
            if use_amp and scaler is not None:
                with torch.autocast(device_type="cuda"):
                    loss_f, loss_u_top, loss_u_rest, loss_v, loss_total = compute_loss(
                        model, x_f, x_bc, u_bc, v_bc, nu, weights,
                        magnetic_case=magnetic_case, H0=H0, chi=chi, D=D,
                    )
                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_f, loss_u_top, loss_u_rest, loss_v, loss_total = compute_loss(
                    model, x_f, x_bc, u_bc, v_bc, nu, weights,
                    magnetic_case=magnetic_case, H0=H0, chi=chi, D=D,
                )
                loss_total.backward()
                optimizer.step()

        # --- Atualização adaptativa dos pesos ---
        if adaptive and epoch > 0 and epoch % grad_every == 0:
            if not isinstance(optimizer, torch.optim.LBFGS):
                x_f_grad = x_f.detach().requires_grad_(True)
                lf, lut, lur, lv, _ = compute_loss(
                    model, x_f_grad, x_bc, u_bc, v_bc, nu, weights,
                    magnetic_case=magnetic_case, H0=H0, chi=chi, D=D,
                )
                losses_dict = {
                    "w_f":      lf,
                    "w_u_top":  lut,
                    "w_u_rest": lur,
                    "w_v":      lv,
                }
                weights = update_weights(model, losses_dict, weights)

        # --- Troca de otimizador Adam → LBFGS ---
        if switch_opt and epoch == 5000:
            optimizer = torch.optim.LBFGS(
                model.parameters(), lr=0.5, max_iter=500,
                history_size=50, line_search_fn="strong_wolfe",
            )

        # --- Resampling de pontos interiores ---
        if resample_every > 0 and epoch > 0 and epoch % resample_every == 0:
            x_f = generate_interior(p["N_int"], p["use_lhs"], device).detach()

        # --- Estatísticas na tampa ---
        with torch.no_grad():
            out_tampa = model(pts_tampa)
            u_avg = float(out_tampa[:, 0].mean().cpu())
            u_std = float(out_tampa[:, 0].std().cpu())
            v_avg = float(out_tampa[:, 1].mean().cpu())
            v_std = float(out_tampa[:, 1].std().cpu())

        # --- Log ---
        row = [
            loss_f.item(), loss_u_top.item(), loss_u_rest.item(),
            loss_v.item(), loss_total.item(),
            u_avg, u_std, v_avg, v_std,
            weights["w_f"], weights["w_u_top"], weights["w_u_rest"], weights["w_v"],
        ]
        if magnetic_case:
            row += [H0, chi, Mn]

        loss_log.append(row)

        log_buffer.append(f"{epoch} " + " ".join(f"{val:.4e}" for val in row) + "\n")
        if len(log_buffer) >= log_flush:
            log_file.writelines(log_buffer)
            log_file.flush()
            log_buffer.clear()

        # --- Print periódico ---
        if epoch % 50 == 0:
            loss_bc_val = loss_u_top.item() + loss_u_rest.item() + loss_v.item()
            w_str = (
                f" | w=[{weights['w_f']:.2f},{weights['w_u_top']:.2f},"
                f"{weights['w_u_rest']:.2f},{weights['w_v']:.2f}]"
                if adaptive else ""
            )
            mn_str = f" | Mn: {Mn:.4e}" if magnetic_case else ""
            print(
                f"  Epoca {epoch:05d} | "
                f"loss_f: {loss_f.item():.4e} | "
                f"loss_bc: {loss_bc_val:.4e} | "
                f"total: {loss_total.item():.4e} | "
                f"<u> tampa: {u_avg:.4f}"
                f"{w_str}{mn_str}"
            )

    # Flush final do log
    if log_buffer:
        log_file.writelines(log_buffer)
    log_file.close()

    # -------------------------------------------------------------------------
    # 8. Pós-processamento
    # -------------------------------------------------------------------------
    loss_log_np = np.array(loss_log)
    plot_losses(loss_log_np, nome_caso, magnetic_case=magnetic_case)
    export_results(
        model, device, nome_caso,
        magnetic_case=magnetic_case, H0=H0, chi=chi, D=D,
    )
    torch.save(model.state_dict(), f"{nome_caso}/model.pt")

    # -------------------------------------------------------------------------
    # 9. Info de execução
    # -------------------------------------------------------------------------
    tempo_total = time.time() - tempo_inicio
    save_execution_info(
        nome_caso, p, tempo_total, weights, device,
        use_compile=p["use_compile"], use_amp=use_amp,
    )
    print(f"  Concluido em {tempo_total:.2f}s -> {nome_caso}/")

    resultado = {
        "output_dir":    nome_caso,
        "loss_final":    round(loss_log[-1][4], 6),
        "u_avg_final":   round(loss_log[-1][5], 6),
        "tempo_total_s": round(tempo_total, 2),
    }
    if magnetic_case:
        resultado["Mn"] = round(Mn, 6)
    return resultado


# ==============================================================================
# PONTO DE ENTRADA
# ==============================================================================
def main():
    CONFIG_PADRAO = "sweep_config.json"

    # Determina o arquivo de configuração
    if len(sys.argv) >= 2 and not sys.argv[1].startswith("--"):
        caminho = sys.argv[1]
    else:
        caminho = CONFIG_PADRAO

    if not os.path.isfile(caminho):
        print(f"Arquivo nao encontrado: {caminho}")
        print("Uso: python main.py [sweep_config.json] [--single]")
        sys.exit(1)

    # --single → simulação única
    if "--single" in sys.argv:
        with open(caminho, "r") as f:
            params = json.load(f)
        resultado = run_simulation(params)
        print("\nResultado:")
        print(json.dumps(resultado, indent=4))
        return

    # Modo sweep: verifica se o JSON tem a chave "runs"
    with open(caminho, "r") as f:
        config = json.load(f)

    if "runs" in config:
        # Varredura de parâmetros
        run_sweep(config, run_simulation)
    else:
        # JSON de simulação única sem --single (conveniência)
        resultado = run_simulation(config)
        print("\nResultado:")
        print(json.dumps(resultado, indent=4))


if __name__ == "__main__":
    main()

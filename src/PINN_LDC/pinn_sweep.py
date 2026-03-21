"""
pinn_sweep.py
=============
Solver PINN para a cavidade cisalhante (lid-driven cavity) com suporte
a varredura de parâmetros (sweep) via arquivo JSON de configuração.

Resolve as equações de Navier-Stokes 2D incompressíveis via
Physics-Informed Neural Network (PINN).

------------------------------------------------------------------------------
USO
------------------------------------------------------------------------------
  Varredura (sweep):
      python pinn_sweep.py sweep_config.json

  Simulação única:
      python pinn_sweep.py params.json  --single

------------------------------------------------------------------------------
DEPENDÊNCIAS
------------------------------------------------------------------------------
  Obrigatórias : torch, numpy, matplotlib, pyvista, psutil
  Opcional     : smt  (LHS real)  →  pip install smt

------------------------------------------------------------------------------
ESTRUTURA DO ARQUIVO
------------------------------------------------------------------------------
  PARTE 1 — SOLVER PINN         (linhas ~90  – ~390)
    - build_net()
    - generate_interior()
    - generate_bc()
    - compute_loss()
    - export_results()
    - plot_losses()
    - run_simulation()

  PARTE 2 — ORQUESTRADOR SWEEP  (linhas ~400 – ~500)
    - carregar_config()
    - montar_params()
    - main()
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import json
import os
import sys
import time
import platform
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psutil
import pyvista as pv

# LHS real via smt (compatível com Python 3.13+)
# Fallback automático para amostragem uniforme se não instalado
try:
    from smt.sampling_methods import LHS as _SMT_LHS
    _HAS_LHS = True
except (ImportError, ModuleNotFoundError):
    _HAS_LHS = False


# ==============================================================================
#
#   ██████   █████  ██████  ████████ ███████      ██
#   ██   ██ ██   ██ ██   ██    ██    ██           ███
#   ██████  ███████ ██████     ██    █████         ██
#   ██      ██   ██ ██   ██    ██    ██            ██
#   ██      ██   ██ ██   ██    ██    ███████       ██
#
#   PARTE 1 — SOLVER PINN
#   Cavidade cisalhante · Navier-Stokes 2D incompressível
#
# ==============================================================================

# ------------------------------------------------------------------------------
# Mapeamento de funções de ativação (string → classe torch)
# ------------------------------------------------------------------------------
ACTIVATIONS = {
    "Tanh":    nn.Tanh,
    "ReLU":    nn.ReLU,
    "Sigmoid": nn.Sigmoid,
    "GELU":    nn.GELU,
}


# ------------------------------------------------------------------------------
# Construção da rede neural
# ------------------------------------------------------------------------------
def build_net(n_layers: int, neurons: int, act_class, use_norm: bool) -> nn.Sequential:
    """Constrói a rede MLP com LayerNorm opcional entre camadas."""
    modules = []
    for i in range(n_layers):
        modules.append(nn.Linear(2 if i == 0 else neurons, neurons))
        if use_norm:
            modules.append(nn.LayerNorm(neurons))
        modules.append(act_class())
    modules.append(nn.Linear(neurons, 3))   # saída: u, v, p
    return nn.Sequential(*modules)


# ------------------------------------------------------------------------------
# Geração de pontos de colocação interiores
# ------------------------------------------------------------------------------
def generate_interior(N: int, use_lhs: bool, device: torch.device) -> torch.Tensor:
    """
    Gera N pontos no domínio [0,1]².
    LHS real (smt) se disponível; uniforme simples como fallback.
    """
    if use_lhs:
        if _HAS_LHS:
            sampling = _SMT_LHS(xlimits=np.array([[0.0, 1.0], [0.0, 1.0]]))
            pts = sampling(N)
        else:
            print("  [aviso] smt não encontrado — usando amostragem uniforme. "
                  "Instale com: pip install smt")
            pts = np.random.uniform(0, 1, (N, 2))
    else:
        pts = np.random.uniform(0, 1, (N, 2))
    return torch.tensor(pts, dtype=torch.float32, device=device)


# ------------------------------------------------------------------------------
# Geração de pontos de contorno e condições de Dirichlet
# ------------------------------------------------------------------------------
def generate_bc(N: int, device: torch.device):
    """
    Retorna (x_bc, u_bc, v_bc) para as quatro paredes da cavidade.
    Tampa superior (y=1): u=1, v=0
    Demais paredes:       u=0, v=0
    """
    lin   = torch.linspace(0, 1, N, device=device).view(-1, 1)
    zeros = torch.zeros_like(lin)
    ones  = torch.ones_like(lin)

    x_bc = torch.cat([
        torch.cat([lin,   zeros], dim=1),   # bottom  (y=0)
        torch.cat([lin,   ones],  dim=1),   # top     (y=1, tampa móvel)
        torch.cat([zeros, lin],   dim=1),   # left    (x=0)
        torch.cat([ones,  lin],   dim=1),   # right   (x=1)
    ], dim=0)

    u_bc = torch.cat([zeros, ones, zeros, zeros], dim=0).view(-1, 1)
    v_bc = torch.zeros_like(u_bc)

    return x_bc, u_bc, v_bc


# ------------------------------------------------------------------------------
# Função de perda — resíduo N-S + condições de contorno
# ------------------------------------------------------------------------------
def compute_loss(model, x_f, x_bc, u_bc, v_bc, nu, weights):
    """
    Calcula perda total = w_f * residuo_NS + w_u_top * erro_tampa
                        + w_u_rest * erro_paredes + w_v * erro_v

    Gradientes de 1ª ordem: 1 passo vetorizado por variável (era 2)
    Gradientes de 2ª ordem: 2 passos extras — total de 5 passes (era 7)
    """
    w_f      = weights["w_f"]
    w_u_top  = weights["w_u_top"]
    w_u_rest = weights["w_u_rest"]
    w_v      = weights["w_v"]

    # --- Pontos interiores: resíduo das equações de N-S ---
    x_f = x_f.clone().requires_grad_(True)
    out = model(x_f)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    def grad1(f):
        """Gradiente de f em relação a x_f — retorna tensor (N, 2)."""
        return torch.autograd.grad(
            f, x_f,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
            retain_graph=True,
        )[0]

    # Derivadas de 1ª ordem (1 passo vetorizado por variável)
    u_xy = grad1(u)
    v_xy = grad1(v)
    p_xy = grad1(p)
    u_x, u_y = u_xy[:, 0:1], u_xy[:, 1:2]
    v_x, v_y = v_xy[:, 0:1], v_xy[:, 1:2]
    p_x, p_y = p_xy[:, 0:1], p_xy[:, 1:2]

    # Derivadas de 2ª ordem (apenas as necessárias para N-S)
    u_xx = grad1(u_x)[:, 0:1]   # ∂²u/∂x²
    u_yy = grad1(u_y)[:, 1:2]   # ∂²u/∂y²
    v_xx = grad1(v_x)[:, 0:1]   # ∂²v/∂x²
    v_yy = grad1(v_y)[:, 1:2]   # ∂²v/∂y²

    # Resíduos: momentum x, momentum y, continuidade
    f_u = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    f_v = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    f_c = u_x + v_y

    loss_f = (f_u**2 + f_v**2 + f_c**2).mean()

    # --- Condições de contorno ---
    out_bc  = model(x_bc)
    u_pred  = out_bc[:, 0:1]
    v_pred  = out_bc[:, 1:2]

    top_mask = (x_bc[:, 1] == 1.0)

    loss_u_top  = nn.MSELoss()(u_pred[top_mask],  u_bc[top_mask])
    loss_u_rest = nn.MSELoss()(u_pred[~top_mask], u_bc[~top_mask])
    loss_v_all  = nn.MSELoss()(v_pred, v_bc)

    loss_bc = w_u_top * loss_u_top + w_u_rest * loss_u_rest + w_v * loss_v_all
    total   = w_f * loss_f + loss_bc

    return loss_f, loss_u_top, loss_u_rest, loss_v_all, total


# ------------------------------------------------------------------------------
# Exportação de resultados: campos, gráficos e VTK
# ------------------------------------------------------------------------------
def export_results(model, device, nome_caso: str):
    """Gera PNGs dos campos u, v, p, linhas de corrente e arquivo VTK."""
    print("  Gerando campos, gráficos e VTK...")
    N = 100
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    XY   = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])

    with torch.no_grad():
        out = model(torch.tensor(XY, dtype=torch.float32, device=device))
        u = out[:, 0].cpu().numpy().reshape(N, N)
        v = out[:, 1].cpu().numpy().reshape(N, N)
        p = out[:, 2].cpu().numpy().reshape(N, N)

    for data, name, cmap in zip(
        [u, v, p], ["u", "v", "pressao"], ["viridis", "viridis", "coolwarm"]
    ):
        plt.contourf(X, Y, data, 50, cmap=cmap)
        plt.colorbar()
        plt.title(f"Campo de {name}")
        plt.axis("scaled")
        plt.savefig(f"{nome_caso}/campo_{name}.png", dpi=300)
        plt.close()

    plt.streamplot(X, Y, u, v, density=1.5, color=np.sqrt(u**2 + v**2), cmap="plasma")
    plt.colorbar()
    plt.title("Linhas de Corrente")
    plt.axis("scaled")
    plt.savefig(f"{nome_caso}/streamlines.png", dpi=300)
    plt.close()

    Xg, Yg, Zg = np.meshgrid(x, y, [0], indexing="ij")
    grid = pv.StructuredGrid()
    grid.points     = np.c_[Xg.ravel(), Yg.ravel(), Zg.ravel()]
    grid.dimensions = [N, N, 1]
    grid.point_data["velocity"] = np.c_[
        u.flatten(order="F"), v.flatten(order="F"), np.zeros_like(u.flatten())
    ]
    grid.point_data["pressure"] = p.flatten(order="F")
    grid.save(f"{nome_caso}/saida_pinn.vtk")

    return u, v, p


# ------------------------------------------------------------------------------
# Gráficos de evolução das perdas
# ------------------------------------------------------------------------------
def plot_losses(loss_log: np.ndarray, nome_caso: str):
    labels = ["loss_f", "loss_bc_u_top", "loss_bc_u_rest", "loss_bc_v", "total"]
    for i, name in enumerate(labels):
        plt.plot(loss_log[:, i], label=name)
    plt.yscale("log")
    plt.legend()
    plt.title("Evolução da função de perda")
    plt.xlabel("Épocas")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{nome_caso}/loss_detalhada.png")
    plt.close()

    for col, var in zip([5, 6], ["u_avg", "u_std"]):
        plt.plot(loss_log[:, col])
        plt.title(f"Evolução de {var} (tampa)")
        plt.xlabel("Épocas")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{nome_caso}/{var}.png")
        plt.close()


# ------------------------------------------------------------------------------
# Função principal do solver — chamada pelo orquestrador ou diretamente
# ------------------------------------------------------------------------------
def run_simulation(params: dict) -> dict:
    """
    Executa uma simulação PINN completa a partir de um dicionário de parâmetros.

    Parâmetros obrigatórios:
        Re, N_int, N_bc, epochs, layers, neurons,
        activation, use_lhs, switch_opt, use_norm,
        w_f, w_u_top, w_u_rest, w_v

    Parâmetros opcionais:
        output_dir      (str)   pasta de saída  (padrão: gerada por timestamp)
        seed            (int)   semente         (padrão: 42)
        use_compile     (bool)  torch.compile   (padrão: True)
        use_amp         (bool)  mixed precision (padrão: True, só GPU)
        resample_every  (int)   resampling x_f  (padrão: 500, 0=desativa)
        log_flush_every (int)   flush do .dat   (padrão: 500)

    Retorna dict com métricas finais.
    """

    # --- Leitura de parâmetros ---
    Re         = float(params["Re"])
    N_int      = int(params["N_int"])
    N_bc       = int(params["N_bc"])
    epochs     = int(params["epochs"])
    n_layers   = int(params["layers"])
    neurons    = int(params["neurons"])
    use_lhs    = bool(params.get("use_lhs",    False))
    switch_opt = bool(params.get("switch_opt", False))
    use_norm   = bool(params.get("use_norm",   False))

    seed            = int(params.get("seed",            42))
    use_compile     = bool(params.get("use_compile",    True))
    use_amp         = bool(params.get("use_amp",        True))
    resample_every  = int(params.get("resample_every",  500))
    log_flush_every = int(params.get("log_flush_every", 500))

    act_name  = params.get("activation", "Tanh")
    act_class = ACTIVATIONS.get(act_name, nn.Tanh)

    weights = {
        "w_f":      float(params.get("w_f",      1.0)),
        "w_u_top":  float(params.get("w_u_top",  1.0)),
        "w_u_rest": float(params.get("w_u_rest", 1.0)),
        "w_v":      float(params.get("w_v",      1.0)),
    }

    # --- Seed reprodutível ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Pasta de saída ---
    if "output_dir" in params:
        nome_caso = params["output_dir"]
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_caso = f"Re{int(Re)}_N{N_int}_B{N_bc}_E{epochs}_{timestamp}"
    os.makedirs(nome_caso, exist_ok=True)

    # --- Dispositivo e viscosidade ---
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = use_amp and (device.type == "cuda")   # AMP só faz sentido em GPU
    nu      = 1.0 / Re

    tempo_inicio = time.time()
    print(f"  Dispositivo: {device} | AMP: {use_amp} | seed: {seed}")

    # --- Salva parâmetros da simulação ---
    with open(f"{nome_caso}/parametros.json", "w") as f:
        json.dump({**params, "activation": act_name, "device": str(device)}, f, indent=4)

    # --- Modelo ---
    model = build_net(n_layers, neurons, act_class, use_norm).to(device)

    if use_compile:
        try:
            model = torch.compile(model)
            print("  torch.compile ativado")
        except Exception:
            print("  torch.compile não disponível (requer PyTorch >= 2.0)")

    # --- Dados ---
    x_f              = generate_interior(N_int, use_lhs, device)
    x_bc, u_bc, v_bc = generate_bc(N_bc, device)

    # Pontos da tampa pré-alocados fora do loop
    pts_tampa = torch.cat(
        [torch.linspace(0, 1, N_bc, device=device).view(-1, 1),
         torch.ones(N_bc, 1, device=device)], dim=1
    )

    # --- Otimizador e AMP scaler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler    = torch.amp.GradScaler("cuda") if use_amp else None

    # --- Log em memória (flush periódico para reduzir I/O) ---
    loss_log   = []
    log_buffer = []
    log_file   = open(f"{nome_caso}/parametros_numericos.dat", "w")
    log_file.write(
        "#epoch loss_f loss_bc_u_top loss_bc_u_rest loss_bc_v loss_total "
        "u_avg u_std v_avg v_std\n"
    )

    # -------------------------------------------------------------------------
    # Loop de treinamento
    # -------------------------------------------------------------------------
    for epoch in range(epochs):
        model.train()

        if isinstance(optimizer, torch.optim.LBFGS):
            def closure():
                optimizer.zero_grad()
                _, _, _, _, loss_c = compute_loss(model, x_f, x_bc, u_bc, v_bc, nu, weights)
                loss_c.backward()
                return loss_c
            optimizer.step(closure)
            with torch.no_grad():
                loss_f, loss_u_top, loss_u_rest, loss_v, loss_total = compute_loss(
                    model, x_f.clone().requires_grad_(True), x_bc, u_bc, v_bc, nu, weights
                )
        else:
            optimizer.zero_grad()
            if use_amp and scaler is not None:
                with torch.autocast(device_type="cuda"):
                    loss_f, loss_u_top, loss_u_rest, loss_v, loss_total = compute_loss(
                        model, x_f, x_bc, u_bc, v_bc, nu, weights
                    )
                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_f, loss_u_top, loss_u_rest, loss_v, loss_total = compute_loss(
                    model, x_f, x_bc, u_bc, v_bc, nu, weights
                )
                loss_total.backward()
                optimizer.step()

        # Troca Adam → LBFGS na época definida
        if switch_opt and epoch == 5000:
            optimizer = torch.optim.LBFGS(
                model.parameters(), lr=0.5, max_iter=500,
                history_size=50, line_search_fn="strong_wolfe",
            )

        # Resampling de pontos interiores
        if resample_every > 0 and epoch > 0 and epoch % resample_every == 0:
            x_f = generate_interior(N_int, use_lhs, device)

        # Estatísticas na tampa
        with torch.no_grad():
            out_tampa = model(pts_tampa)
            u_avg = float(out_tampa[:, 0].mean().cpu())
            u_std = float(out_tampa[:, 0].std().cpu())
            v_avg = float(out_tampa[:, 1].mean().cpu())
            v_std = float(out_tampa[:, 1].std().cpu())

        row = [
            loss_f.item(), loss_u_top.item(), loss_u_rest.item(),
            loss_v.item(), loss_total.item(),
            u_avg, u_std, v_avg, v_std,
        ]
        loss_log.append(row)

        log_buffer.append(f"{epoch} " + " ".join(f"{val:.4e}" for val in row) + "\n")
        if len(log_buffer) >= log_flush_every:
            log_file.writelines(log_buffer)
            log_file.flush()
            log_buffer.clear()

        if epoch % 50 == 0:
            loss_bc_val = loss_u_top.item() + loss_u_rest.item() + loss_v.item()
            print(
                f"  Época {epoch:05d} | "
                f"loss_f: {loss_f.item():.4e} | "
                f"loss_bc: {loss_bc_val:.4e} | "
                f"total: {loss_total.item():.4e} | "
                f"<u> tampa: {u_avg:.4f}"
            )

    if log_buffer:
        log_file.writelines(log_buffer)
    log_file.close()

    # -------------------------------------------------------------------------
    # Pós-processamento
    # -------------------------------------------------------------------------
    loss_log_np = np.array(loss_log)
    plot_losses(loss_log_np, nome_caso)
    export_results(model, device, nome_caso)
    torch.save(model.state_dict(), f"{nome_caso}/model.pt")

    # -------------------------------------------------------------------------
    # Informações de hardware
    # -------------------------------------------------------------------------
    tempo_total = time.time() - tempo_inicio
    with open(f"{nome_caso}/info_execucao.json", "w") as f:
        json.dump({
            "tempo_total_segundos": round(tempo_total, 2),
            "cpu":               platform.processor(),
            "arquitetura":       platform.machine(),
            "sistema":           platform.system() + " " + platform.release(),
            "cpu_cores_fisicos": psutil.cpu_count(logical=False),
            "cpu_cores_logicos": psutil.cpu_count(logical=True),
            "memoria_total_GB":  round(psutil.virtual_memory().total / 1e9, 2),
            "gpu_disponivel":    torch.cuda.is_available(),
            "nome_gpu":          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nenhuma",
            "torch_compile":     use_compile,
            "amp":               use_amp,
            "seed":              seed,
        }, f, indent=4)

    print(f"  Concluido em {tempo_total:.2f}s -> {nome_caso}/")

    return {
        "output_dir":    nome_caso,
        "loss_final":    round(loss_log[-1][4], 6),
        "u_avg_final":   round(loss_log[-1][5], 6),
        "tempo_total_s": round(tempo_total, 2),
    }


# ==============================================================================
#
#   ██████   █████  ██████  ████████ ███████     ██████
#   ██   ██ ██   ██ ██   ██    ██    ██               ██
#   ██████  ███████ ██████     ██    █████         ████
#   ██      ██   ██ ██   ██    ██    ██           ██
#   ██      ██   ██ ██   ██    ██    ███████      ███████
#
#   PARTE 2 — ORQUESTRADOR DE VARREDURA (SWEEP)
#   Lê sweep_config.json e executa run_simulation() para cada run
#
# ==============================================================================

# ------------------------------------------------------------------------------
# Leitura do arquivo de configuração
# ------------------------------------------------------------------------------
def carregar_config(caminho: str) -> dict:
    with open(caminho, "r") as f:
        return json.load(f)


# ------------------------------------------------------------------------------
# Mescla parâmetros base com overrides do run individual
# ------------------------------------------------------------------------------
def montar_params(base: dict, run: dict) -> dict:
    """O run sobrescreve apenas os campos que define — o resto vem do base."""
    params = base.copy()
    params.update(run)
    return params


# ------------------------------------------------------------------------------
# Ponto de entrada principal
# ------------------------------------------------------------------------------
def main():
    # JSON padrão — usado quando nenhum argumento é passado
    CONFIG_PADRAO = "sweep_config.json"

    caminho = sys.argv[1] if len(sys.argv) >= 2 and not sys.argv[1].startswith("--")               else CONFIG_PADRAO

    if not os.path.isfile(caminho):
        print(f"Arquivo não encontrado: {caminho}")
        print("Uso: python pinn_sweep.py [sweep_config.json] [--single]")
        sys.exit(1)

    # Modo simulação única
    if "--single" in sys.argv:
        with open(caminho, "r") as f:
            params = json.load(f)
        resultado = run_simulation(params)
        print("\nResultado:")
        print(json.dumps(resultado, indent=4))
        return

    # Modo varredura
    config     = carregar_config(caminho)
    sweep_name = config.get("sweep_name", "sweep")
    base       = config["base"]
    runs       = config["runs"]
    n_total    = len(runs)

    raiz = Path(sweep_name)
    raiz.mkdir(exist_ok=True)

    # Salva cópia do JSON usado (rastreabilidade)
    with open(raiz / "sweep_config_usado.json", "w") as f:
        json.dump(config, f, indent=4)

    print(f"\n{'='*60}")
    print(f"  Varredura : {sweep_name}")
    print(f"  Total runs: {n_total}")
    print(f"  Saida em  : {raiz}/")
    print(f"{'='*60}\n")

    resultados     = []
    tempo_varredura = time.time()

    for i, run in enumerate(runs, start=1):
        run_id = run.get("run_id", f"run_{i:02d}")
        params = montar_params(base, run)
        params["output_dir"] = str(raiz / run_id)

        print(f"[{i}/{n_total}] {run_id} | "
              f"w_f={params.get('w_f')} | w_u_top={params.get('w_u_top')} | "
              f"w_u_rest={params.get('w_u_rest')} | w_v={params.get('w_v')}")

        t0 = time.time()
        try:
            resultado              = run_simulation(params)
            resultado["run_id"]    = run_id
            resultado["status"]    = "ok"
            resultado["tempo_run_s"] = round(time.time() - t0, 2)
            print(f"    ok — {resultado['tempo_run_s']:.1f}s\n")
        except Exception as e:
            resultado = {
                "run_id": run_id, "status": "erro",
                "mensagem": str(e), "tempo_run_s": round(time.time() - t0, 2)
            }
            print(f"    erro em {run_id}: {e}\n")

        resultados.append(resultado)

    tempo_total = time.time() - tempo_varredura

    with open(raiz / "resumo_varredura.json", "w") as f:
        json.dump({
            "sweep_name":   sweep_name,
            "n_runs":       n_total,
            "tempo_total_s": round(tempo_total, 2),
            "runs":         resultados,
        }, f, indent=4)

    print(f"{'='*60}")
    print(f"  Varredura concluida em {tempo_total:.1f}s")
    print(f"  Resumo: {raiz}/resumo_varredura.json")
    print(f"{'='*60}\n")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    main()

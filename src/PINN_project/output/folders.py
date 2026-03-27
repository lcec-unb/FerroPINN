"""
output/folders.py
=================
Criação da pasta de saída, geração do nome do caso e gerenciamento de logs.

Funções públicas
----------------
    make_output_dir(params_loaded, output_dir_override) -> str
        Cria a pasta de saída e retorna o caminho.
        O nome embute "mag" se magnetic_case=True.

    save_params_json(nome_caso, params_loaded) -> None
        Salva o JSON de parâmetros da simulação na pasta de saída.

    open_log(nome_caso, adaptive_weights, grad_update_every,
             opt_name, opt_lr, switch_opt, magnetic_case) -> file
        Abre o arquivo .dat de log e escreve o cabeçalho.

    save_execution_info(nome_caso, params_loaded, tempo_total,
                        weights, device, use_compile, use_amp) -> None
        Salva info_execucao.json com hardware e métricas finais.
"""

import json
import os
import platform
import time
from datetime import datetime

import psutil
import torch


# ------------------------------------------------------------------------------
# Nome e criação da pasta de saída
# ------------------------------------------------------------------------------
def make_output_dir(params_loaded: dict, output_dir_override: str = None) -> str:
    """
    Determina e cria a pasta de saída da simulação.

    Prioridade:
      1. output_dir_override (passado pelo sweep)
      2. params_loaded["output_dir"] (do JSON)
      3. Nome gerado automaticamente com timestamp

    O nome gerado embute "mag" se magnetic_case=True, e "nomag" caso contrário.

    Retorna o caminho da pasta criada.
    """
    if output_dir_override:
        nome_caso = output_dir_override
    elif params_loaded.get("output_dir"):
        nome_caso = params_loaded["output_dir"]
    else:
        timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
        Re            = int(params_loaded["Re"])
        N_int         = params_loaded["N_int"]
        epochs        = params_loaded["epochs"]
        magnetic_case = params_loaded["magnetic_case"]

        if magnetic_case:
            H0  = params_loaded["H0"]
            chi = params_loaded["chi"]
            tag = f"mag_H{int(H0)}_chi{chi}"
        else:
            tag = "nomag"

        nome_caso = f"Re{Re}_{tag}_N{N_int}_E{epochs}_{timestamp}"

    os.makedirs(nome_caso, exist_ok=True)
    return nome_caso


# ------------------------------------------------------------------------------
# Salvar parâmetros da simulação
# ------------------------------------------------------------------------------
def save_params_json(nome_caso: str, params_loaded: dict) -> None:
    """
    Salva parametros.json na pasta de saída com os valores usados na simulação.
    Campos internos (prefixo _) e objetos não serializáveis são omitidos.
    """
    raw = params_loaded.get("_raw", {})

    extra = {
        "activation": params_loaded["act_name"],
        "optimizer":  params_loaded["opt_name"],
        "lr":         params_loaded["opt_lr"],
        "nu":         params_loaded["nu"],
    }

    if params_loaded["magnetic_case"] and params_loaded["Mn"] is not None:
        extra["Mn_calculado"] = round(params_loaded["Mn"], 8)

    with open(f"{nome_caso}/parametros.json", "w") as f:
        json.dump({**raw, **extra}, f, indent=4)


# ------------------------------------------------------------------------------
# Abertura do arquivo de log numérico
# ------------------------------------------------------------------------------
def open_log(
    nome_caso:        str,
    adaptive_weights: bool,
    grad_update_every: int,
    opt_name:         str,
    opt_lr:           float,
    switch_opt:       bool,
    magnetic_case:    bool,
):
    """
    Abre parametros_numericos.dat e escreve o cabeçalho.
    Retorna o objeto file aberto (deve ser fechado pelo chamador).

    Colunas do .dat:
      epoch | loss_f | loss_u_top | loss_u_rest | loss_v | loss_total
            | u_avg  | u_std      | v_avg       | v_std
            | w_f    | w_u_top    | w_u_rest    | w_v
            | [H0 | chi | Mn]  ← somente se magnetic_case=True
    """
    log_file = open(f"{nome_caso}/parametros_numericos.dat", "w")

    meta = (
        f"# optimizer={opt_name} lr={opt_lr} switch_opt={switch_opt} "
        f"adaptive_weights={adaptive_weights} "
        f"grad_update_every={grad_update_every} "
        f"magnetic_case={magnetic_case}\n"
    )

    if magnetic_case:
        header = (
            "#epoch loss_f loss_bc_u_top loss_bc_u_rest loss_bc_v loss_total "
            "u_avg u_std v_avg v_std w_f w_u_top w_u_rest w_v H0 chi Mn\n"
        )
    else:
        header = (
            "#epoch loss_f loss_bc_u_top loss_bc_u_rest loss_bc_v loss_total "
            "u_avg u_std v_avg v_std w_f w_u_top w_u_rest w_v\n"
        )

    log_file.write(meta)
    log_file.write(header)
    return log_file


# ------------------------------------------------------------------------------
# Informações de hardware e execução
# ------------------------------------------------------------------------------
def save_execution_info(
    nome_caso:    str,
    params_loaded: dict,
    tempo_total:  float,
    weights:      dict,
    device:       torch.device,
    use_compile:  bool,
    use_amp:      bool,
) -> None:
    """
    Salva info_execucao.json com informações de hardware, hiperparâmetros
    e pesos finais. Inclui campos magnéticos apenas se magnetic_case=True.
    """
    info = {
        "tempo_total_segundos": round(tempo_total, 2),
        "cpu":               platform.processor(),
        "arquitetura":       platform.machine(),
        "sistema":           platform.system() + " " + platform.release(),
        "cpu_cores_fisicos": psutil.cpu_count(logical=False),
        "cpu_cores_logicos": psutil.cpu_count(logical=True),
        "memoria_total_GB":  round(psutil.virtual_memory().total / 1e9, 2),
        "gpu_disponivel":    torch.cuda.is_available(),
        "nome_gpu":          (torch.cuda.get_device_name(0)
                              if torch.cuda.is_available() else "Nenhuma"),
        "torch_compile":         use_compile,
        "amp":                   use_amp,
        "seed":                  params_loaded["seed"],
        "optimizer":             params_loaded["opt_name"],
        "lr_inicial":            params_loaded["opt_lr"],
        "switch_opt":            params_loaded["switch_opt"],
        "adaptive_weights":      params_loaded["adaptive_weights"],
        "grad_update_every":     params_loaded["grad_update_every"],
        "w_f_final":             round(weights["w_f"],      6),
        "w_u_top_final":         round(weights["w_u_top"],  6),
        "w_u_rest_final":        round(weights["w_u_rest"], 6),
        "w_v_final":             round(weights["w_v"],      6),
        "magnetic_case":         params_loaded["magnetic_case"],
    }

    # Campos magnéticos — só incluídos se magnetic_case=True
    if params_loaded["magnetic_case"]:
        info["H0"]  = params_loaded["H0"]
        info["chi"] = params_loaded["chi"]
        info["D"]   = params_loaded["D"]
        if params_loaded["Mn"] is not None:
            info["Mn"] = round(params_loaded["Mn"], 8)

    with open(f"{nome_caso}/info_execucao.json", "w") as f:
        json.dump(info, f, indent=4)

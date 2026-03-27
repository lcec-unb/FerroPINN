"""
config/parameters.py
====================
Leitura e validação de todos os parâmetros do JSON de entrada.
Também inicializa a seed reprodutível e o dispositivo (CPU/GPU).

Funções públicas
----------------
    load_params(params: dict) -> dict
        Lê o dicionário bruto do JSON e retorna um dicionário tipado e
        completo, com todos os valores padrão preenchidos.

    setup_seed(seed: int) -> None
        Inicializa as seeds do Python, NumPy e PyTorch (CPU + GPU).

    get_device(use_amp: bool) -> tuple[torch.device, bool]
        Retorna o dispositivo e o flag AMP corrigido (só True se GPU).

Parâmetros suportados no JSON
------------------------------
  Obrigatórios:
    Re, N_int, N_bc, epochs, layers, neurons

  Opcionais (com padrão):
    activation      str     "Tanh"
    optimizer       str     "Adam"
    lr              float   1e-3
    use_lhs         bool    false
    switch_opt      bool    false
    use_norm        bool    false
    use_compile     bool    true
    use_amp         bool    true
    resample_every  int     500
    log_flush_every int     500
    seed            int     42
    w_f             float   1.0
    w_u_top         float   1.0
    w_u_rest        float   1.0
    w_v             float   1.0
    adaptive_weights     bool  false
    grad_update_every    int   100

  Caso magnético (lidos apenas se magnetic_case=true):
    magnetic_case   bool    false
    H0              float   —  (obrigatório se magnetic_case=true)
    chi             float   —  (obrigatório se magnetic_case=true)
    D               float   1.0
"""

import numpy as np
import torch
import torch.nn as nn

# ------------------------------------------------------------------------------
# Mapeamentos string → classe torch
# ------------------------------------------------------------------------------
ACTIVATIONS = {
    "Tanh":    nn.Tanh,
    "ReLU":    nn.ReLU,
    "Sigmoid": nn.Sigmoid,
    "GELU":    nn.GELU,
}

OPTIMIZERS = {
    "Adam":    torch.optim.Adam,
    "AdamW":   torch.optim.AdamW,
    "RAdam":   torch.optim.RAdam,
    "NAdam":   torch.optim.NAdam,
    "RMSprop": torch.optim.RMSprop,
    "SGD":     torch.optim.SGD,
}

# Permeabilidade magnética do vácuo (usada somente no caso magnético)
MU0 = 4.0 * np.pi * 1e-7


# ------------------------------------------------------------------------------
# Leitura e validação de parâmetros
# ------------------------------------------------------------------------------
def load_params(params: dict) -> dict:
    """
    Recebe o dicionário bruto do JSON e retorna um dicionário completo
    com todos os valores tipados e padrões preenchidos.

    Levanta ValueError se parâmetros obrigatórios estiverem ausentes.
    """
    # Obrigatórios
    for key in ("Re", "N_int", "N_bc", "epochs", "layers", "neurons"):
        if key not in params:
            raise ValueError(f"Parâmetro obrigatório ausente no JSON: '{key}'")

    magnetic_case = bool(params.get("magnetic_case", False))

    # Parâmetros magnéticos — validados apenas se magnetic_case=true
    if magnetic_case:
        for key in ("H0", "chi"):
            if key not in params:
                raise ValueError(
                    f"magnetic_case=true mas parâmetro obrigatório ausente: '{key}'"
                )
        H0  = float(params["H0"])
        chi = float(params["chi"])
        D   = float(params.get("D", 1.0))
        Mn  = MU0 * chi * H0**2 * D
    else:
        H0  = None
        chi = None
        D   = None
        Mn  = None

    act_name  = params.get("activation", "Tanh")
    opt_name  = params.get("optimizer",  "Adam")

    if act_name not in ACTIVATIONS:
        raise ValueError(f"Ativação desconhecida: '{act_name}'. "
                         f"Opções: {list(ACTIVATIONS)}")
    if opt_name not in OPTIMIZERS:
        raise ValueError(f"Otimizador desconhecido: '{opt_name}'. "
                         f"Opções: {list(OPTIMIZERS)}")

    return {
        # Físicos
        "Re":    float(params["Re"]),
        "nu":    1.0 / float(params["Re"]),

        # Domínio / amostragem
        "N_int":   int(params["N_int"]),
        "N_bc":    int(params["N_bc"]),
        "use_lhs": bool(params.get("use_lhs", False)),

        # Rede neural
        "layers":    int(params["layers"]),
        "neurons":   int(params["neurons"]),
        "act_name":  act_name,
        "act_class": ACTIVATIONS[act_name],
        "use_norm":  bool(params.get("use_norm", False)),

        # Treinamento
        "epochs":        int(params["epochs"]),
        "opt_name":      opt_name,
        "opt_class":     OPTIMIZERS[opt_name],
        "opt_lr":        float(params.get("lr", 1e-3)),
        "switch_opt":    bool(params.get("switch_opt",   False)),
        "use_compile":   bool(params.get("use_compile",  True)),
        "use_amp":       bool(params.get("use_amp",      True)),
        "resample_every":  int(params.get("resample_every",  500)),
        "log_flush_every": int(params.get("log_flush_every", 500)),
        "seed":            int(params.get("seed", 42)),

        # Pesos da loss
        "weights": {
            "w_f":      float(params.get("w_f",      1.0)),
            "w_u_top":  float(params.get("w_u_top",  1.0)),
            "w_u_rest": float(params.get("w_u_rest", 1.0)),
            "w_v":      float(params.get("w_v",      1.0)),
        },

        # Pesos adaptativos
        "adaptive_weights":  bool(params.get("adaptive_weights",  False)),
        "grad_update_every": int(params.get("grad_update_every", 100)),

        # Caso magnético
        "magnetic_case": magnetic_case,
        "H0":  H0,
        "chi": chi,
        "D":   D,
        "Mn":  Mn,

        # Pasta de saída (pode ser sobrescrita pelo sweep)
        "output_dir": params.get("output_dir", None),

        # Cópia do JSON bruto (para salvar na pasta de saída)
        "_raw": params,
    }


# ------------------------------------------------------------------------------
# Seed reprodutível
# ------------------------------------------------------------------------------
def setup_seed(seed: int) -> None:
    """Inicializa seeds do NumPy e PyTorch (CPU e GPU)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------------------
# Dispositivo e AMP
# ------------------------------------------------------------------------------
def get_device(use_amp: bool) -> tuple:
    """
    Detecta CPU/GPU e corrige o flag AMP (AMP só funciona em GPU).
    Retorna (device, use_amp_corrected).
    """
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = use_amp and (device.type == "cuda")
    return device, use_amp

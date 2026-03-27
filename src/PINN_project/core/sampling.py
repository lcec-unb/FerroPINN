"""
core/sampling.py
================
Geração de pontos de colocação interiores e de contorno (BCs Dirichlet).

Funções públicas
----------------
    generate_interior(N, use_lhs, device) -> Tensor  shape (N, 2)
        Pontos no domínio [0,1]² via LHS (smt) ou amostragem uniforme.

    generate_bc(N, device) -> (x_bc, u_bc, v_bc)
        Pontos nas 4 paredes com condições de velocidade Dirichlet.

Condições de contorno
---------------------
  Tampa superior (y=1) : u = 1,  v = 0   ← lid em movimento
  Demais paredes       : u = 0,  v = 0   ← no-slip
"""

import numpy as np
import torch

# LHS real via smt — fallback automático para uniforme se não instalado
try:
    from smt.sampling_methods import LHS as _SMT_LHS
    _HAS_LHS = True
except (ImportError, ModuleNotFoundError):
    _HAS_LHS = False


# ------------------------------------------------------------------------------
# Pontos interiores
# ------------------------------------------------------------------------------
def generate_interior(
    N:       int,
    use_lhs: bool,
    device:  torch.device,
) -> torch.Tensor:
    """
    Gera N pontos no domínio [0,1]².

    Se use_lhs=True e smt estiver instalado, usa Latin Hypercube Sampling
    para melhor cobertura do domínio. Caso contrário, usa amostragem
    aleatória uniforme.

    Retorna Tensor float32 de shape (N, 2) sem requires_grad
    (o requires_grad é ativado dentro de compute_loss).
    """
    if use_lhs:
        if _HAS_LHS:
            sampling = _SMT_LHS(xlimits=np.array([[0.0, 1.0], [0.0, 1.0]]))
            pts = sampling(N)
        else:
            print("  [aviso] smt nao encontrado — usando amostragem uniforme. "
                  "Instale com: pip install smt")
            pts = np.random.uniform(0, 1, (N, 2))
    else:
        pts = np.random.uniform(0, 1, (N, 2))

    return torch.tensor(pts, dtype=torch.float32, device=device)


# ------------------------------------------------------------------------------
# Pontos de contorno
# ------------------------------------------------------------------------------
def generate_bc(
    N:      int,
    device: torch.device,
):
    """
    Gera pontos nas 4 paredes da cavidade com condições Dirichlet.

    Retorna
    -------
    x_bc : Tensor (4N, 2) — coordenadas (x, y) dos pontos de contorno
    u_bc : Tensor (4N, 1) — valor alvo da componente u
    v_bc : Tensor (4N, 1) — valor alvo da componente v (sempre 0)

    Ordem das paredes: bottom → top → left → right
    """
    lin   = torch.linspace(0, 1, N, device=device).view(-1, 1)
    zeros = torch.zeros_like(lin)
    ones  = torch.ones_like(lin)

    x_bc = torch.cat([
        torch.cat([lin,   zeros], dim=1),   # bottom  y=0
        torch.cat([lin,   ones],  dim=1),   # top     y=1  (tampa móvel)
        torch.cat([zeros, lin],   dim=1),   # left    x=0
        torch.cat([ones,  lin],   dim=1),   # right   x=1
    ], dim=0)

    # Tampa: u=1 | demais: u=0
    u_bc = torch.cat([zeros, ones, zeros, zeros], dim=0).view(-1, 1)
    v_bc = torch.zeros_like(u_bc)

    return x_bc, u_bc, v_bc

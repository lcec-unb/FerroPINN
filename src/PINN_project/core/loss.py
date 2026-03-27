"""
core/loss.py
============
Função de perda PINN para a cavidade cisalhante (lid-driven cavity).

Suporta dois modos controlados pelo flag `magnetic_case`:

  magnetic_case=False  →  Navier-Stokes 2D incompressível padrão
  magnetic_case=True   →  N-S com força de Kelvin (ferrofluido)
                           f_v += Fmag = -Mn * (1 - y/D)
                           Mn  = mu0 * chi * H0^2 * D

Funções públicas
----------------
    compute_loss(model, x_f, x_bc, u_bc, v_bc, nu, weights,
                 magnetic_case, H0, chi, D)
        -> (loss_f, loss_u_top, loss_u_rest, loss_v, loss_total)

Derivadas
---------
  1ª ordem  : 1 passo vetorizado por variável (u, v, p)
  2ª ordem  : calculadas apenas onde necessário (u_xx, u_yy, v_xx, v_yy)
  Total     : 7 backward passes (mínimo necessário para N-S 2D)
"""

import torch
import torch.nn as nn
import numpy as np

# Permeabilidade magnética do vácuo
MU0 = 4.0 * np.pi * 1e-7


# ------------------------------------------------------------------------------
# Helper: gradiente de primeira ordem vetorizado
# ------------------------------------------------------------------------------
def _grad1(f, x):
    """
    Retorna o gradiente de f em relação a x.
    f : Tensor (N, 1)
    x : Tensor (N, 2) com requires_grad=True
    Retorna Tensor (N, 2)  →  [:, 0] = df/dx,  [:, 1] = df/dy
    """
    return torch.autograd.grad(
        f, x,
        grad_outputs=torch.ones_like(f),
        create_graph=True,
        retain_graph=True,
    )[0]


# ------------------------------------------------------------------------------
# Função de perda principal
# ------------------------------------------------------------------------------
def compute_loss(
    model,
    x_f,
    x_bc,
    u_bc,
    v_bc,
    nu:            float,
    weights:       dict,
    magnetic_case: bool  = False,
    H0:            float = None,
    chi:           float = None,
    D:             float = None,
):
    """
    Calcula as perdas do PINN para N-S 2D (com ou sem campo magnético).

    Parâmetros
    ----------
    model          : rede neural PyTorch
    x_f            : Tensor (N_int, 2) — pontos interiores (sem requires_grad,
                     clonado internamente)
    x_bc           : Tensor (4*N_bc, 2) — pontos de contorno
    u_bc, v_bc     : Tensor (4*N_bc, 1) — valores alvo das BCs
    nu             : viscosidade cinemática (= 1/Re)
    weights        : dict com w_f, w_u_top, w_u_rest, w_v
    magnetic_case  : se True, adiciona força de Kelvin em f_v
    H0, chi, D     : parâmetros magnéticos (necessários se magnetic_case=True)

    Retorna
    -------
    loss_f, loss_u_top, loss_u_rest, loss_v_all, loss_total
    Todos são scalares (Tensor 0-d) com grafo computacional intacto.
    """
    w_f      = weights["w_f"]
    w_u_top  = weights["w_u_top"]
    w_u_rest = weights["w_u_rest"]
    w_v      = weights["w_v"]

    # ------------------------------------------------------------------
    # Resíduo das equações de Navier-Stokes nos pontos interiores
    # ------------------------------------------------------------------
    x_f = x_f.clone().requires_grad_(True)
    out = model(x_f)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    # Derivadas de 1ª ordem (1 passo vetorizado por variável)
    u_xy = _grad1(u, x_f)
    v_xy = _grad1(v, x_f)
    p_xy = _grad1(p, x_f)

    u_x, u_y = u_xy[:, 0:1], u_xy[:, 1:2]
    v_x, v_y = v_xy[:, 0:1], v_xy[:, 1:2]
    p_x, p_y = p_xy[:, 0:1], p_xy[:, 1:2]

    # Derivadas de 2ª ordem (apenas as necessárias para o Laplaciano)
    u_xx = _grad1(u_x, x_f)[:, 0:1]   # ∂²u/∂x²
    u_yy = _grad1(u_y, x_f)[:, 1:2]   # ∂²u/∂y²
    v_xx = _grad1(v_x, x_f)[:, 0:1]   # ∂²v/∂x²
    v_yy = _grad1(v_y, x_f)[:, 1:2]   # ∂²v/∂y²

    # Resíduos: momentum x, continuidade (comuns aos dois casos)
    f_u = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    f_c = u_x + v_y

    # Resíduo momentum y — com ou sem força de Kelvin
    if magnetic_case:
        # Força de Kelvin: Fmag = -Mn * (1 - y/D)
        # Mn = mu0 * chi * H0^2 * D
        Mn   = MU0 * chi * H0**2 * D
        y    = x_f[:, 1:2]
        Fmag = -Mn * (1.0 - y / D)
        f_v  = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy) + Fmag
    else:
        f_v = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    loss_f = (f_u**2 + f_v**2 + f_c**2).mean()

    # ------------------------------------------------------------------
    # Condições de contorno
    # ------------------------------------------------------------------
    out_bc = model(x_bc)
    u_pred = out_bc[:, 0:1]
    v_pred = out_bc[:, 1:2]

    # Separar tampa (y=1) das demais paredes
    top_mask = (x_bc[:, 1] == 1.0)

    loss_u_top  = nn.MSELoss()(u_pred[top_mask],  u_bc[top_mask])
    loss_u_rest = nn.MSELoss()(u_pred[~top_mask], u_bc[~top_mask])
    loss_v_all  = nn.MSELoss()(v_pred, v_bc)

    # ------------------------------------------------------------------
    # Loss total ponderada
    # ------------------------------------------------------------------
    loss_bc = w_u_top * loss_u_top + w_u_rest * loss_u_rest + w_v * loss_v_all
    total   = w_f * loss_f + loss_bc

    return loss_f, loss_u_top, loss_u_rest, loss_v_all, total

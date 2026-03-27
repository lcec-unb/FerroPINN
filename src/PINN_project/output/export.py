"""
output/export.py
================
Exportação de resultados pós-processados: campos de velocidade/pressão,
campo magnético H (caso magnético), linhas de corrente, VTK e gráficos
de evolução das perdas e pesos adaptativos.

Funções públicas
----------------
    export_results(model, device, nome_caso,
                   magnetic_case, H0, chi, D) -> None
        Gera PNGs dos campos u, v, p [e H], streamlines e arquivo VTK.

    plot_losses(loss_log, nome_caso, magnetic_case) -> None
        Gera loss_detalhada.png, u_avg.png, u_std.png [e pesos_adaptativos.png].
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyvista as pv


# Permeabilidade magnética (para cálculo do campo H)
MU0 = 4.0 * np.pi * 1e-7

# Resolução do grid de pós-processamento
_GRID_N = 100


# ------------------------------------------------------------------------------
# Campo H analítico (caso magnético)
# ------------------------------------------------------------------------------
def _compute_H_field(Y: np.ndarray, H0: float, D: float) -> np.ndarray:
    """
    H(x, y) = H0 * (1 - y/D)
    Campo uniforme em x, linear em y (aplicado externamente).
    Y : array meshgrid de coordenadas y, shape (N, N).
    """
    return H0 * (1.0 - Y / D)


# ------------------------------------------------------------------------------
# Exportação dos campos e VTK
# ------------------------------------------------------------------------------
def export_results(
    model,
    device:        torch.device,
    nome_caso:     str,
    magnetic_case: bool  = False,
    H0:            float = None,
    chi:           float = None,
    D:             float = None,
) -> None:
    """
    Avalia a rede no grid de pós-processamento e salva:
      - campo_u.png, campo_v.png, campo_pressao.png
      - campo_H.png                    ← somente se magnetic_case=True
      - streamlines.png
      - saida_pinn.vtk  (inclui H_magnetic se magnetic_case=True)

    Parâmetros
    ----------
    model          : rede neural PyTorch (em modo eval)
    device         : dispositivo usado no treinamento
    nome_caso      : pasta de saída
    magnetic_case  : se True, calcula e exporta o campo H
    H0, chi, D     : parâmetros magnéticos (necessários se magnetic_case=True)
    """
    print("  Gerando campos, graficos e VTK...")
    N = _GRID_N
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    XY   = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])

    # Campos u, v, p via rede neural
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(XY, dtype=torch.float32, device=device))
        u = out[:, 0].cpu().numpy().reshape(N, N)
        v = out[:, 1].cpu().numpy().reshape(N, N)
        p = out[:, 2].cpu().numpy().reshape(N, N)

    # PNGs — campos de velocidade e pressão
    campos_base = [
        (u, "u",      "viridis",  "Campo de u (velocidade horizontal)"),
        (v, "v",      "viridis",  "Campo de v (velocidade vertical)"),
        (p, "pressao","coolwarm", "Campo de pressao"),
    ]

    # Campo H adicional (caso magnético)
    if magnetic_case and H0 is not None:
        H = _compute_H_field(Y, H0, D)
        campos_base.append(
            (H, "H", "plasma",
             f"Campo magnetico H  (H0={H0:.1f}, chi={chi})")
        )
    else:
        H = None

    for data, name, cmap, titulo in campos_base:
        plt.contourf(X, Y, data, 50, cmap=cmap)
        plt.colorbar()
        plt.title(titulo)
        plt.axis("scaled")
        plt.savefig(f"{nome_caso}/campo_{name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Linhas de corrente
    speed = np.sqrt(u**2 + v**2)
    plt.streamplot(X, Y, u, v, density=1.5, color=speed, cmap="plasma")
    plt.colorbar()
    plt.title("Linhas de Corrente")
    plt.axis("scaled")
    plt.savefig(f"{nome_caso}/streamlines.png", dpi=300, bbox_inches="tight")
    plt.close()

    # VTK
    Xg, Yg, Zg = np.meshgrid(x, y, [0], indexing="ij")
    grid = pv.StructuredGrid()
    grid.points     = np.c_[Xg.ravel(), Yg.ravel(), Zg.ravel()]
    grid.dimensions = [N, N, 1]
    grid.point_data["velocity"] = np.c_[
        u.flatten(order="F"),
        v.flatten(order="F"),
        np.zeros_like(u.flatten()),
    ]
    grid.point_data["pressure"] = p.flatten(order="F")
    if magnetic_case and H is not None:
        grid.point_data["H_magnetic"] = H.flatten(order="F")
    grid.save(f"{nome_caso}/saida_pinn.vtk")


# ------------------------------------------------------------------------------
# Gráficos de evolução das perdas e pesos
# ------------------------------------------------------------------------------
def plot_losses(
    loss_log:      np.ndarray,
    nome_caso:     str,
    magnetic_case: bool = False,
) -> None:
    """
    Gera gráficos de evolução a partir de loss_log (numpy array 2D).

    Colunas esperadas:
      0  loss_f          4  loss_total
      1  loss_u_top      5  u_avg
      2  loss_u_rest     6  u_std
      3  loss_v          7  v_avg
                         8  v_std
      9  w_f            10  w_u_top
      11 w_u_rest       12  w_v
      [13 H0  14 chi  15 Mn]  ← somente se magnetic_case=True

    Gera:
      - loss_detalhada.png
      - u_avg.png, u_std.png
      - pesos_adaptativos.png  (se colunas de peso presentes)
    """
    labels = ["loss_f", "loss_bc_u_top", "loss_bc_u_rest", "loss_bc_v", "total"]
    for i, name in enumerate(labels):
        plt.plot(loss_log[:, i], label=name)
    plt.yscale("log")
    plt.legend()
    plt.title("Evolucao da funcao de perda")
    plt.xlabel("Epocas")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{nome_caso}/loss_detalhada.png")
    plt.close()

    # u_avg e u_std na tampa
    for col, var in zip([5, 6], ["u_avg", "u_std"]):
        plt.plot(loss_log[:, col])
        plt.title(f"Evolucao de {var} (tampa)")
        plt.xlabel("Epocas")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{nome_caso}/{var}.png")
        plt.close()

    # Pesos adaptativos (colunas 9–12)
    if loss_log.shape[1] >= 13:
        w_labels = ["w_f", "w_u_top", "w_u_rest", "w_v"]
        for col, name in zip([9, 10, 11, 12], w_labels):
            plt.plot(loss_log[:, col], label=name)
        plt.title("Evolucao dos pesos adaptativos")
        plt.xlabel("Epocas")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{nome_caso}/pesos_adaptativos.png")
        plt.close()

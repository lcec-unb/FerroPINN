import pyvista as pv
import numpy as np

# -----------------------------
# 1. Ler arquivos
# -----------------------------
foam = pv.read("internal.vtu")
pinn = pv.read("saida_pinn.vtk")


print("FOAM arrays:", foam.array_names)
print("PINN arrays:", pinn.array_names)

print("FOAM point data:", foam.point_data.keys())
print("FOAM cell data :", foam.cell_data.keys())

print("PINN point data:", pinn.point_data.keys())
print("PINN cell data :", pinn.cell_data.keys())

# -----------------------------
# 2. Garantir dados em pontos
# -----------------------------
if len(foam.point_data) == 0:
    foam = foam.cell_data_to_point_data()

if len(pinn.point_data) == 0:
    pinn = pinn.cell_data_to_point_data()


# -----------------------------
# 3. Interpolar PINN → OpenFOAM
# -----------------------------
pinn_interp = foam.interpolate(pinn)

# -----------------------------
# 4. Erro de pressão
# -----------------------------
p_foam = foam["p"]
p_pinn = pinn_interp["p"]

error_p = np.abs(p_foam - p_pinn)
foam["error_p"] = error_p

# -----------------------------
# 5. Erro de velocidade (norma)
# -----------------------------
U_foam = foam["U"]
U_pinn = pinn_interp["U"]

error_U = np.linalg.norm(U_foam - U_pinn, axis=1)
foam["error_U"] = error_U

# -----------------------------
# 6. Plotar e salvar imagens
# -----------------------------

# --- pressão ---
plotter_p = pv.Plotter(off_screen=True)
plotter_p.add_mesh(
    foam,
    scalars="error_p",
    cmap="inferno",
    show_edges=False
)
plotter_p.add_scalar_bar(title="|p_OF - p_PINN|")
plotter_p.view_xy()   # bom para casos 2D
plotter_p.screenshot("error_pressure.png")
plotter_p.close()

# --- velocidade ---
plotter_u = pv.Plotter(off_screen=True)
plotter_u.add_mesh(
    foam,
    scalars="error_U",
    cmap="viridis",
    show_edges=False
)
plotter_u.add_scalar_bar(title="||U_OF - U_PINN||")
plotter_u.view_xy()
plotter_u.screenshot("error_velocity.png")
plotter_u.close()

print("Imagens geradas:")
print(" - error_pressure.png")
print(" - error_velocity.png")

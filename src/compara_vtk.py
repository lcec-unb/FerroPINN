import numpy as np
import pyvista as pv

# --- arquivos ---
vtk_of = "internal.vtu"      # saída do OpenFOAM
vtk_pinn = "saida_pinn.vtk"  # saída do PINN

# --- leitura ---
of = pv.read(vtk_of)
pinn = pv.read(vtk_pinn)

# garantir que as variáveis existam
U_of = of.point_data["U"]
p_of = of.point_data["p"]

# nomes no PINN (ajustar se diferentes)
U_pinn = pinn.point_data["velocity"]
p_pinn = pinn.point_data["pressure"]

# --- resample: interpolar PINN para os pontos do OpenFOAM ---
resampled = of.sample(pinn)

U_pinn_interp = resampled.point_data["velocity"]
p_pinn_interp = resampled.point_data["pressure"]

# --- calcular diferenças ---
dU = np.abs(U_of - U_pinn_interp)
dp = np.abs(p_of - p_pinn_interp)

# erros velocidade por componente
mean_U_comp = np.mean(dU)
max_U_comp = np.max(dU)

# erros velocidade módulo
dU_mag = np.linalg.norm(U_of - U_pinn_interp, axis=1)
mean_U_mag = np.mean(dU_mag)
max_U_mag = np.max(dU_mag)

# erros pressão
mean_p = np.mean(dp)
max_p = np.max(dp)

# --- saída limpa ---
print("== Estatísticas de erro absoluto ==")
print(f"Velocidade | por componente:  médio = {mean_U_comp:.6e} | máximo = {max_U_comp:.6e}")
print(f"Velocidade | módulo (||ΔU||):  médio = {mean_U_mag:.6e} | máximo = {max_U_mag:.6e}")
print(f"Pressão    | absoluto:  médio = {mean_p:.6e} | máximo = {max_p:.6e}")


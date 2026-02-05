import pyvista as pv
import numpy as np

# =====================================================
# 1. Configurações e Carregamento
# =====================================================
print("Lendo arquivos...")
foam = pv.read("internal.vtu")
pinn = pv.read("saida_pinn.vtk")

foam = foam.cell_data_to_point_data()
pinn = pinn.cell_data_to_point_data()

# =====================================================
# 2. Alinhamento Geométrico
# =====================================================
foam.points[:, 2] = 0.0
pinn.points[:, 2] = 0.0

width_foam = foam.bounds[1] - foam.bounds[0]
width_pinn = pinn.bounds[1] - pinn.bounds[0]
scale_factor = width_foam / width_pinn

if abs(scale_factor - 1.0) > 1e-4:
    print(f"Ajustando escala da PINN. Fator: {scale_factor:.4f}")
    pinn.points *= scale_factor

# =====================================================
# 3. Interpolação
# =====================================================
xmin, xmax, ymin, ymax, _, _ = foam.bounds
nx, ny = 300, 300 

refined = pv.ImageData(
    dimensions=(nx, ny, 1),
    spacing=((xmax - xmin)/(nx-1), (ymax - ymin)/(ny-1), 1.0),
    origin=(xmin, ymin, 0.0)
)

print("Interpolando campos...")
radius_search = (xmax - xmin) / nx * 2.0
foam_r = refined.interpolate(foam, radius=radius_search)
pinn_r = refined.interpolate(pinn, radius=radius_search)

# =====================================================
# 4. Processamento da Física
# =====================================================
pinn_vel_key = [k for k in pinn.point_data.keys() if k.lower() in ['u', 'velocity', 'vel']][0]
u_foam = foam_r["U"]
u_pinn = pinn_r[pinn_vel_key]

mag_u_foam = np.linalg.norm(u_foam[:, :2], axis=1)
mag_u_pinn = np.linalg.norm(u_pinn[:, :2], axis=1)
error_u_abs = np.abs(mag_u_foam - mag_u_pinn)

p_foam_raw = foam_r["p"]
p_pinn_key = [k for k in pinn.point_data.keys() if k.lower() in ['p', 'pressure']][0]
p_pinn_raw = pinn_r[p_pinn_key]

p_foam_centered = p_foam_raw - np.nanmean(p_foam_raw)
p_pinn_centered = p_pinn_raw - np.nanmean(p_pinn_raw)
error_p_abs = np.abs(p_foam_centered - p_pinn_centered)

refined["error_u_abs"] = error_u_abs
refined["error_p_abs"] = error_p_abs

# =====================================================
# 5. Geração de Imagens (Correção AttributeError)
# =====================================================
def salvar_mapa_erro_limpo(grid, scalar_name, filename, title, cmap):
    # Janela larga para acomodar a legenda à direita
    p = pv.Plotter(off_screen=True, window_size=[1400, 1000])
    p.background_color = "white"
    
    vmax = np.percentile(grid[scalar_name], 98)
    
    sargs = dict(
        title=title,
        title_font_size=22,
        label_font_size=18,
        fmt="%.2e",
        color='black',
        vertical=True,
        position_x=0.85, # Bem à direita
        position_y=0.1,
        height=0.8
    )
    
    p.add_mesh(
        grid, 
        scalars=scalar_name, 
        cmap=cmap, 
        clim=[0, vmax],
        lighting=False,
        scalar_bar_args=sargs,
        show_edges=False
    )
    
    p.view_xy()
    
    # CORREÇÃO AQUI: Usando o método mais robusto do PyVista
    p.enable_parallel_projection() 
    
    # Resetar a câmera para focar no objeto antes do screenshot
    p.reset_camera()
    # Pequeno ajuste de zoom para dar margem
    p.camera.zoom(0.85)
    
    p.screenshot(filename)
    p.close()
    print(f"Gerado: {filename} (Escala: 0 a {vmax:.2e})")

# Execução
salvar_mapa_erro_limpo(refined, "error_u_abs", "erro_velocidade.png", "Erro Absoluto U", "jet")
salvar_mapa_erro_limpo(refined, "error_p_abs", "erro_pressao.png", "Erro Absoluto P", "inferno")

refined.save("comparacao_final.vti")
print("\nProcesso finalizado com sucesso!")
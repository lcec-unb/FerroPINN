import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os, json, shutil
import pyvista as pv
from datetime import datetime
import time
import platform
import psutil

# -----------------------------
# Função de construção da rede
# -----------------------------
def build_net(layers, neurons, act, norm):
    modules = []
    for i in range(layers):
        modules.append(nn.Linear(2 if i == 0 else neurons, neurons))
        if norm: 
            modules.append(nn.LayerNorm(neurons))
        modules.append(act())
    modules.append(nn.Linear(neurons, 3))
    return nn.Sequential(*modules)

# -----------------------------
# Funções auxiliares
# -----------------------------
def generate_points(N, tipo="uniforme", device="cpu"):
    return torch.tensor(
        np.random.uniform(0, 1, (N, 2)) if tipo == "lhs" else np.random.rand(N, 2),
        dtype=torch.float32
    ).to(device)

def generate_bc(N, device="cpu"):
    lin = torch.linspace(0, 1, N, device=device).view(-1, 1)
    bottom = torch.cat([lin, torch.zeros_like(lin)], dim=1)
    top = torch.cat([lin, torch.ones_like(lin)], dim=1)
    left = torch.cat([torch.zeros_like(lin), lin], dim=1)
    right = torch.cat([torch.ones_like(lin), lin], dim=1)

    x_bc = torch.cat([bottom, top, left, right], dim=0)
    u_bc = torch.cat([torch.zeros_like(lin), torch.ones_like(lin), torch.zeros_like(lin), torch.zeros_like(lin)], dim=0)
    v_bc = torch.zeros_like(u_bc)
    return x_bc, u_bc.view(-1,1), v_bc.view(-1,1)

def gradients(u, x, order=1):
    for _ in range(order):
        u = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    return u

# -----------------------------
# Função principal de simulação
# -----------------------------
def rodar_simulacao(params, indice):
    (
        Re, N_int, N_bc, epochs, layers, neurons, activation, 
        use_lhs, switch_opt, use_norm, w_f, w_u_top, w_u_rest, w_v
    ) = params

    tempo_inicio = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nu = 1.0 / Re

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_tmp = f"Re{int(Re)}_N{N_int}_B{N_bc}_E{epochs}_{timestamp}_sim{indice}"
    os.makedirs(nome_tmp, exist_ok=True)

    # Salvar parâmetros no JSON
    with open(f"{nome_tmp}/parametros.json", "w") as f:
        json.dump({
            "Re": Re, "N_int": N_int, "N_bc": N_bc, "epochs": epochs,
            "layers": layers, "neurons": neurons, "activation": activation.__name__,
            "LHS": use_lhs, "Troca_Opt_5000": switch_opt, "Normalizacao": use_norm,
            "w_f": w_f, "w_u_top": w_u_top, "w_u_rest": w_u_rest, "w_v": w_v
        }, f, indent=4)

    # Modelo
    model = build_net(layers, neurons, activation, use_norm).to(device)

    # Dados
    x_f = generate_points(N_int, "lhs" if use_lhs else "uniforme", device)
    x_bc, u_bc, v_bc = generate_bc(N_bc, device)

    # Função de perda
    def loss_function(model, x_f, x_bc, u_bc, v_bc):
        x_f.requires_grad = True
        out_f = model(x_f)
        u, v, p = out_f[:, 0:1], out_f[:, 1:2], out_f[:, 2:3]

        grads = lambda f: gradients(f, x_f)
        u_x, u_y = grads(u)[:, 0:1], grads(u)[:, 1:2]
        v_x, v_y = grads(v)[:, 0:1], grads(v)[:, 1:2]
        p_x, p_y = grads(p)[:, 0:1], grads(p)[:, 1:2]
        u_xx, u_yy = gradients(u_x, x_f)[:, 0:1], gradients(u_y, x_f)[:, 1:2]
        v_xx, v_yy = gradients(v_x, x_f)[:, 0:1], gradients(v_y, x_f)[:, 1:2]

        f_u = u*u_x + v*u_y + p_x - nu*(u_xx + u_yy)
        f_v = u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)
        f_c = u_x + v_y
        loss_f = (f_u**2 + f_v**2 + f_c**2).mean()

        out_bc = model(x_bc)
        u_pred, v_pred = out_bc[:, 0:1], out_bc[:, 1:2]
        top_idx = (x_bc[:,1] == 1.0).squeeze()
        non_top_idx = ~top_idx
        all_idx = torch.arange(x_bc.shape[0])

        loss_bc_u_top = nn.MSELoss()(u_pred[top_idx], u_bc[top_idx])
        loss_bc_u_rest = nn.MSELoss()(u_pred[non_top_idx], u_bc[non_top_idx])
        loss_bc_v = nn.MSELoss()(v_pred[all_idx], v_bc[all_idx])

        loss_bc = w_u_top * loss_bc_u_top + w_u_rest * loss_bc_u_rest + w_v * loss_bc_v
        total = w_f * loss_f + loss_bc

        return loss_f, loss_bc_u_top, loss_bc_u_rest, loss_bc_v, total

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_log = []

    # Arquivo de log
    log_file = open(f"{nome_tmp}/parametros_numericos.dat", "w")
    log_file.write("#epoch loss_f loss_bc_u_top loss_bc_u_rest loss_bc_v u_avg u_std v_avg v_std\n")

    for epoch in range(epochs):
        model.train()
        loss_f, loss_bc_u_top, loss_bc_u_rest, loss_bc_v, loss_total = loss_function(model, x_f, x_bc, u_bc, v_bc)
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # estatísticas na tampa
        x_sup = torch.linspace(0, 1, N_bc, device=device).view(-1, 1)
        y_sup = torch.ones_like(x_sup)
        pts = torch.cat([x_sup, y_sup], dim=1)
        with torch.no_grad():
            u_pred = model(pts)[:, 0].cpu().numpy()
            v_pred = model(pts)[:, 1].cpu().numpy()
            u_avg, u_std = u_pred.mean(), u_pred.std()
            v_avg, v_std = v_pred.mean(), v_pred.std()

        loss_log.append([
            loss_f.item(), loss_bc_u_top.item(), loss_bc_u_rest.item(),
            loss_bc_v.item(), loss_total.item(), u_avg, u_std, v_avg, v_std
        ])
        log_file.write(f"{epoch} {loss_log[-1][0]:.4e} {loss_log[-1][1]:.4e} {loss_log[-1][2]:.4e} {loss_log[-1][3]:.4e} {u_avg:.4e} {u_std:.4e} {v_avg:.4e} {v_std:.4e}\n")

        # print a cada 50 épocas
        if epoch % 50 == 0:
            loss_bc_val = loss_bc_u_top + loss_bc_u_rest + loss_bc_v
            print(f"🧮 Sim {indice} | Época {epoch:05d} | Loss_f: {loss_f.item():.4e} | Loss_bc: {loss_bc_val.item():.4e} | Total: {loss_total.item():.4e} | ⬆️ ⟨u⟩ (tampa): {u_avg:.4f}")

    log_file.close()

    # Gráficos
    loss_log = np.array(loss_log)
    labels = ["loss_f", "loss_bc_u_top", "loss_bc_u_rest", "loss_bc_v", "total"]
    for i, name in enumerate(labels):
        plt.plot(loss_log[:, i], label=name)
    plt.yscale("log")
    plt.legend()
    plt.title("Função de perda - evolução")
    plt.savefig(f"{nome_tmp}/loss_detalhada.png")
    plt.close()

    # u_avg e u_std
    for i, var in zip([5, 6], ["u_avg", "u_std"]):
        plt.plot(loss_log[:, i])
        plt.title(f"Evolução de {var}")
        plt.xlabel("Épocas")
        plt.grid(True)
        plt.savefig(f"{nome_tmp}/{var}.png")
        plt.close()

    # Campos simulados
    N = 100
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    XY = np.hstack([X.reshape(-1,1), Y.reshape(-1,1)])
    with torch.no_grad():
        out = model(torch.tensor(XY, dtype=torch.float32, device=device))
        u = out[:,0].cpu().numpy().reshape(N,N)
        v = out[:,1].cpu().numpy().reshape(N,N)
        p = out[:,2].cpu().numpy().reshape(N,N)

    # PNGs
    for data, name, cmap in zip([u, v, p], ["u", "v", "pressao"], ["viridis", "viridis", "coolwarm"]):
        plt.contourf(X, Y, data, 50, cmap=cmap)
        plt.colorbar()
        plt.title(f"Campo de {name}")
        plt.axis("scaled")
        plt.savefig(f"{nome_tmp}/campo_{name}.png", dpi=300)
        plt.close()

    plt.streamplot(X, Y, u, v, density=1.5, color=np.sqrt(u**2 + v**2), cmap="plasma")
    plt.colorbar()
    plt.title("Linhas de Corrente")
    plt.axis("scaled")
    plt.savefig(f"{nome_tmp}/streamlines.png", dpi=300)
    plt.close()

    # VTK
    grid = pv.StructuredGrid()
    Xg, Yg, Zg = np.meshgrid(x, y, [0], indexing='ij')
    grid.points = np.c_[Xg.ravel(), Yg.ravel(), Zg.ravel()]
    grid.dimensions = [N, N, 1]
    velocity = np.c_[u.flatten(order='F'), v.flatten(order='F'), np.zeros_like(u.flatten())]
    grid.point_data["velocity"] = velocity
    grid.point_data["pressure"] = p.flatten(order='F')
    grid.save(f"{nome_tmp}/saida_pinn.vtk")

    # Infos de execução
    tempo_fim = time.time()
    sistema_info = {
        "tempo_total_segundos": round(tempo_fim - tempo_inicio, 2),
        "cpu": platform.processor(),
        "arquitetura": platform.machine(),
        "sistema": platform.system() + " " + platform.release(),
        "cpu_cores_fisicos": psutil.cpu_count(logical=False),
        "cpu_cores_logicos": psutil.cpu_count(logical=True),
        "memoria_total_GB": round(psutil.virtual_memory().total / 1e9, 2),
        "gpu_disponivel": torch.cuda.is_available(),
        "nome_gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nenhuma"
    }
    with open(f"{nome_tmp}/info_execucao.json", "w") as f:
        json.dump(sistema_info, f, indent=4)

    print(f"✅ Simulação {indice} concluída. Resultados em: {nome_tmp}/")

    return nome_tmp, int(Re)   # retornamos a pasta temporária e o Reynolds


# -----------------------------
# Entrada de múltiplas simulações
# -----------------------------
n_sims = int(input("Quantas simulações deseja rodar? "))

simulacoes = []
for i in range(n_sims):
    print(f"\n--- Parâmetros da Simulação {i+1} ---")
    Re = float(input("🔢 Digite o número de Reynolds: "))
    N_int = int(input("🔢 Nº de pontos internos: "))
    N_bc = int(input("🔢 Nº de pontos de contorno: "))
    epochs = int(input("🔁 Nº de épocas: "))
    layers = int(input("🏗️ Nº de camadas da rede: "))
    neurons = int(input("🧠 Nº de neurônios por camada: "))

    activations = {
        "1": nn.Tanh,
        "2": nn.ReLU,
        "3": nn.Sigmoid,
        "4": nn.GELU
    }
    print("🎚️ Escolha a função de ativação:\n1️⃣  Tanh\n2️⃣  ReLU\n3️⃣  Sigmoid\n4️⃣  GELU")
    act_choice = input("Digite o número: ")
    activation = activations.get(act_choice, nn.Tanh)

    use_lhs = input("📐 Usar LHS? (s/n): ").strip().lower() == "s"
    switch_opt = input("🔁 Trocar otimizador após 5000 épocas? (s/n): ").strip().lower() == "s"
    use_norm = input("🧪 Usar normalização em camadas? (s/n): ").strip().lower() == "s"

    w_f = float(input("⚖️  Peso para o termo do interior: "))
    w_u_top = float(input("⚖️  Peso para u na tampa superior: "))
    w_u_rest = float(input("⚖️  Peso para u nas demais paredes: "))
    w_v = float(input("⚖️  Peso para v em todas as paredes: "))

    simulacoes.append([Re, N_int, N_bc, epochs, layers, neurons, activation, use_lhs, switch_opt, use_norm, w_f, w_u_top, w_u_rest, w_v])

# -----------------------------
# Rodando todas as simulações e organizando em pastas pai
# -----------------------------
resultados = []
for i, params in enumerate(simulacoes, start=1):
    pasta_tmp, Re_val = rodar_simulacao(params, i)
    resultados.append((pasta_tmp, Re_val))

for pasta_tmp, Re_val in resultados:
    pasta_pai = f"Re{Re_val}"  # já criada pelo bash
    destino = os.path.join(pasta_pai, pasta_tmp)
    shutil.move(pasta_tmp, destino)
    print(f"📂 Movido: {pasta_tmp} → {destino}")



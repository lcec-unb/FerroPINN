"""
core/network.py
===============
Construção da rede neural MLP usada como ansatz da solução PINN.

Funções públicas
----------------
    build_net(n_layers, neurons, act_class, use_norm) -> nn.Sequential
        Constrói e retorna o modelo MLP.

Arquitetura
-----------
  Entrada : 2 neurônios  (coordenadas x, y)
  Ocultas : n_layers camadas de `neurons` neurônios cada
            com LayerNorm opcional entre Linear e Ativação
  Saída   : 3 neurônios  (u, v, p)
"""

import torch.nn as nn


def build_net(
    n_layers:  int,
    neurons:   int,
    act_class,
    use_norm:  bool,
) -> nn.Sequential:
    """
    Constrói MLP com LayerNorm opcional.

    Parâmetros
    ----------
    n_layers  : número de camadas ocultas
    neurons   : número de neurônios por camada oculta
    act_class : classe de ativação (ex: nn.Tanh)
    use_norm  : se True, insere LayerNorm após cada Linear

    Retorna
    -------
    nn.Sequential com as camadas montadas.
    """
    modules = []
    for i in range(n_layers):
        in_features = 2 if i == 0 else neurons
        modules.append(nn.Linear(in_features, neurons))
        if use_norm:
            modules.append(nn.LayerNorm(neurons))
        modules.append(act_class())
    modules.append(nn.Linear(neurons, 3))   # saída: u, v, p
    return nn.Sequential(*modules)

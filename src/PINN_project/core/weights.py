"""
core/weights.py
===============
Atualização adaptativa dos pesos da função de perda.

Implementa o método de Gradient-norm Balancing (Wang et al. 2021)
com quatro correções para estabilidade numérica:

  1. FREEZE  — Termos com loss < TOL_FREEZE são congelados.
               Evita que w_f -> inf quando loss_f já convergiu.
  2. CLAMP   — Nenhum peso ultrapassa W_MAX. Proteção extra.
  3. EMA     — Suavização exponencial para evitar picos abruptos.
  4. NORM    — Média dos pesos ativos normalizada para 1.0,
               mantendo a escala total da loss estável.

Constantes configuráveis
------------------------
  W_MAX      : teto absoluto de qualquer peso            (padrão: 100.0)
  EMA_ALPHA  : fator de suavização EMA                   (padrão: 0.2)
  TOL_FREEZE : loss abaixo desse valor → peso congelado  (padrão: 1e-7)

Funções públicas
----------------
    compute_grad_norm(loss, model) -> float
        Norma L2 dos gradientes de `loss` em relação aos pesos do modelo.

    update_weights(model, losses_dict, weights) -> dict
        Recalcula e retorna os pesos atualizados.
"""

import numpy as np
import torch

# ------------------------------------------------------------------------------
# Constantes — ajuste aqui se precisar de comportamento diferente
# ------------------------------------------------------------------------------
W_MAX      = 100.0   # teto absoluto de qualquer peso
EMA_ALPHA  = 0.2     # fator EMA  (0 = sem atualização, 1 = sem memória)
TOL_FREEZE = 1e-7    # loss abaixo desse valor → peso congelado


# ------------------------------------------------------------------------------
# Norma do gradiente de um termo da loss
# ------------------------------------------------------------------------------
def compute_grad_norm(loss: torch.Tensor, model: torch.nn.Module) -> float:
    """
    Calcula a norma L2 dos gradientes de `loss` em relação a todos os
    parâmetros treináveis do modelo.

    Parâmetros
    ----------
    loss  : escalar Tensor com grafo computacional intacto
    model : rede neural PyTorch

    Retorna
    -------
    float — norma L2 dos gradientes
    """
    grads = torch.autograd.grad(
        loss,
        model.parameters(),
        retain_graph=True,
        allow_unused=True,
        create_graph=False,
    )
    norm_sq = sum(g.norm() ** 2 for g in grads if g is not None)
    return norm_sq.sqrt().item()


# ------------------------------------------------------------------------------
# Atualização dos pesos com grad-norm balancing + correções
# ------------------------------------------------------------------------------
def update_weights(
    model,
    losses_dict: dict,
    weights:     dict,
) -> dict:
    """
    Atualiza os pesos com grad-norm balancing + freeze + clamp + EMA + norm.

    Parâmetros
    ----------
    model       : rede neural PyTorch
    losses_dict : dict mapeando cada chave de peso ao seu tensor de loss.
                  Ex: {"w_f": loss_f, "w_u_top": loss_u_top, ...}
                  Os tensores devem ter grafo computacional intacto.
    weights     : dict atual com os valores float dos pesos.

    Retorna
    -------
    dict com os novos pesos (mesmas chaves que `weights`).

    Comportamento por termo
    -----------------------
    - loss < TOL_FREEZE  → peso congelado (mantido sem alteração)
    - loss >= TOL_FREEZE → peso recalculado via grad-norm, depois
                           suavizado com EMA e limitado por W_MAX,
                           por fim normalizado junto com os demais ativos
    """
    # 1. Calcula grad-norms apenas para termos não convergidos
    loss_vals = {k: v.item() for k, v in losses_dict.items()}
    norms = {}
    for key, loss_tensor in losses_dict.items():
        if loss_vals[key] < TOL_FREEZE:
            norms[key] = None    # congelado
        else:
            norms[key] = compute_grad_norm(loss_tensor, model) + 1e-8

    # 2. Média das normas dos termos ativos
    active_norms = [v for v in norms.values() if v is not None]
    if not active_norms:
        return weights   # todos convergidos — nada a fazer

    mean_norm = float(np.mean(active_norms))

    # 3. Pesos-alvo + EMA + clamp por termo
    new_weights = {}
    for key in weights:
        if norms[key] is None:
            # Congelado: mantém peso atual
            new_weights[key] = weights[key]
        else:
            # Alvo proporcional ao déficit de gradiente
            target = mean_norm / norms[key]
            target = float(np.clip(target, 1.0 / W_MAX, W_MAX))
            # EMA: suaviza transições bruscas
            new_weights[key] = (1.0 - EMA_ALPHA) * weights[key] + EMA_ALPHA * target

    # 4. Normaliza para que a média dos pesos ativos seja 1.0
    #    → mantém a escala total da loss estável durante o treinamento
    active_keys = [k for k in new_weights if norms[k] is not None]
    mean_w = float(np.mean([new_weights[k] for k in active_keys]))
    if mean_w > 0:
        for k in active_keys:
            new_weights[k] /= mean_w

    return new_weights

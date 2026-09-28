"""Conjuntos de nivel sobre a entrada: a Gram das posicoes de uma imagem e as suas distancias.

Os objetos sao as posicoes espaciais de uma imagem num bloco (grade de ate `max_side`); as
tres fontes e a composicao sao as do metodo (`src/pixel_energy.py`), e o campo de energia e
lido nessa Gram **com** a intensidade de cada posicao.

As distancias dos conjuntos de nivel sao lidas depois de remover a intensidade:
G̃ = D^{-β/2} G D^{-β/2}, com β = 1 (a transformacao angular T_a) e um piso de 10% da mediana
na diagonal para posicoes de energia quase nula, que nao tem direcao definida. Sem isso, a
distancia a uma referencia e dominada pela energia da outra posicao (medido no Cat/Dog:
Spearman 0,50–0,74 com a energia nos blocos rasos), e as faixas "proximas" caem no fundo.
Normalizacoes parciais (β = 0,5 e 0,75) nao bastaram.

As referencias sao posicoes tipicas, nao extremas: s0 e o medoide (na distancia angular) dos
10% de posicoes de maior energia; s1, o medoide da metade de menor energia.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .pixel_energy import joint_transform, pixel_energy, pixel_grams


def image_gram(src, i: int, block: str, device, max_side: int | None = 32,
               joint: str = "trace_center") -> tuple[torch.Tensor, int, int]:
    """Gram conjunta transformada (1, P, P) das posicoes de uma imagem num bloco."""
    acts, grads = src.acts[block][i : i + 1], src.grads[block][i : i + 1]
    if max_side and acts.shape[-1] > max_side:
        acts = F.adaptive_avg_pool2d(acts, max_side)
        grads = F.adaptive_avg_pool2d(grads, max_side)
    h, w = acts.shape[-2:]
    grams, _ = pixel_grams(acts, grads, src.metric[block], device, ("joint",), "id", 0.0)
    G, _ = joint_transform(grams["joint"], joint)
    return G, h, w


def angular(g: np.ndarray, floor: float = 0.1, beta: float = 1.0) -> np.ndarray:
    """G̃ = D^{-β/2} G D^{-β/2}, com piso na diagonal."""
    if beta == 0:
        return g
    d = np.diag(g)
    scale = np.maximum(d, floor * max(float(np.median(d)), 1e-300)) ** (beta / 2)
    return g / np.outer(scale, scale)


def distances_from(gt: np.ndarray, s: int) -> np.ndarray:
    d = np.diag(gt)
    return np.sqrt(np.maximum(d[s] + d - 2.0 * gt[s], 0.0))


def medoid(gt: np.ndarray, idx: np.ndarray) -> int:
    d = np.diag(gt)
    sub = np.sqrt(np.maximum(d[idx, None] + d[None, idx] - 2.0 * gt[np.ix_(idx, idx)], 0.0))
    return int(idx[sub.sum(1).argmin()])


def readout(G: torch.Tensor, h: int, w: int, knn_fraction: float = 0.05,
            high_fraction: float = 0.1, floor: float = 0.1) -> dict:
    """Campo de energia, referencias, funcoes distancia, vizinhanca N_k(s0) e diagnosticos."""
    energy = pixel_energy(G, "participation", 0.95)["E_par"][0].cpu().numpy()
    g = G[0].cpu().numpy()
    gt = angular(g, floor)
    order = np.argsort(energy)
    k_high = max(2, int(round(high_fraction * len(order))))
    s0 = medoid(gt, order[-k_high:])
    s1 = medoid(gt, order[: len(order) // 2])
    p0, p1 = distances_from(gt, s0), distances_from(gt, s1)
    k = max(1, int(round(knn_fraction * h * w)))
    knn = np.zeros(h * w, dtype=bool)
    knn[np.argsort(p0, kind="stable")[: k + 1]] = True
    rows, cols = np.divmod(np.arange(h * w), w)
    r0, c0 = divmod(s0, w)
    others = np.arange(h * w) != s0
    return {
        "energy": energy.reshape(h, w),
        "p0": p0.reshape(h, w),
        "p1": p1.reshape(h, w),
        "s0": (r0, c0),
        "s1": divmod(s1, w),
        "knn": knn.reshape(h, w),
        "rho_energy": float(spearmanr(p0[others], np.diag(g)[others]).statistic),
        "rho_spatial": float(spearmanr(p0[others], np.hypot(rows - r0, cols - c0)[others]).statistic),
    }


def mask_to_grid(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """Fracao de animal em cada celula da grade (mascara 0/1 na resolucao da imagem)."""
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.adaptive_avg_pool2d(t, (h, w))[0, 0].numpy()


def center_prior(h: int, w: int, sigma: float = 0.25) -> np.ndarray:
    """Um mapa que so marca o centro: o piso para qualquer leitura de localizacao, porque os
    animais desta base estao, em geral, centralizados na foto."""
    rows, cols = np.mgrid[0:h, 0:w]
    r = ((rows + 0.5) / h - 0.5) ** 2 + ((cols + 0.5) / w - 0.5) ** 2
    return np.exp(-r / (2 * sigma**2))

"""Diagnóstico de distâncias entre amostras no espaço RKHS.

Dois regimes de distância, lado a lado:
- Distância RKHS contínua  D[i,j] = ||Φ(xᵢ) - Φ(xⱼ)||
- Cruzamentos de nível      C[i,j] = ceil(D[i,j] / δ)

Ambas construídas a partir do mesmo kernel K_tipos = K_W ⊙ K_A ⊙ K_G via
``sentence_distance_matrix`` e ``level_crossing_matrix``.
"""

from __future__ import annotations

from typing import Any

import torch

from ..gram.distance import level_crossing_matrix, sentence_distance_matrix
from ..pipelines.sample_geometry import class_pair_means


def _pair_matrix(
    means: dict[tuple[str, str], float],
    unique: list[str],
) -> torch.Tensor:
    """Monta matriz simétrica (C, C) de médias por par de categorias."""
    c = len(unique)
    idx = {cat: i for i, cat in enumerate(unique)}
    M = torch.zeros(c, c)
    for (ca, cb), v in means.items():
        i, j = idx[ca], idx[cb]
        M[i, j] = v
        M[j, i] = v
    return M


def plot_sentence_distance_diagnostics(
    sample_kernels: dict[int, torch.Tensor],
    labels: list[str],
    categories: list[str],
    layer_idx: int = 2,
    n_levels: int = 12,
    quantile: float = 0.90,
    figsize: tuple[float, float] = (16, 9),
    suptitle: str = (
        "Distâncias entre frases no espaço de representação do modelo\n"
        "Topo: distância RKHS completa D²=Kii−2Kij+Kjj.  "
        "Base: distância discretizada por curvas de nível."
    ),
) -> Any:
    """Grade 2×3 de diagnósticos de distância entre amostras.

    Linha superior — distância RKHS contínua:
      • Heatmap completo D para ``layer_idx``.
      • Média por par de classes (heatmap anotado).
      • Evolução da distância média por par de classes ao longo das camadas.

    Linha inferior — cruzamentos de nível discretizados:
      • Heatmap de contagem de cruzamentos C para ``layer_idx``.
      • Média de cruzamentos por par de classes.
      • Evolução dos cruzamentos médios ao longo das camadas.

    O delta de discretização é calibrado por camada:
    ``δ = quantile(D_off_diag, q) / n_levels``.

    Parameters
    ----------
    sample_kernels : dict camada → K_tipos (N, N).
    labels         : rótulo de cada amostra (ex.: ``"0:main"``).
    categories     : categoria de cada amostra (para agrupamento de cores).
    layer_idx      : camada usada nos heatmaps da esquerda.
    n_levels       : número de níveis para a discretização.
    quantile       : quantil das distâncias usado para calibrar δ.
    figsize        : tamanho da figura.
    suptitle       : título principal.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np

    unique_cats = sorted(set(categories))
    all_layers = sorted(sample_kernels.keys())

    # ------------------------------------------------------------------
    # Pré-computa D e C para todas as camadas
    # ------------------------------------------------------------------
    dist_by_layer: dict[int, torch.Tensor] = {}
    cross_by_layer: dict[int, torch.Tensor] = {}
    pair_dist_by_layer: dict[int, dict] = {}
    pair_cross_by_layer: dict[int, dict] = {}

    for li in all_layers:
        K = sample_kernels[li]
        D = sentence_distance_matrix(K)
        C = level_crossing_matrix(D, n_levels=n_levels, quantile=quantile)
        dist_by_layer[li] = D
        cross_by_layer[li] = C
        pair_dist_by_layer[li] = class_pair_means(D, categories)
        pair_cross_by_layer[li] = class_pair_means(C, categories)

    # ------------------------------------------------------------------
    # Pares de categorias para as linhas de evolução
    # ------------------------------------------------------------------
    pair_keys: list[tuple[str, str]] = []
    for i, ca in enumerate(unique_cats):
        for cb in unique_cats[i + 1:]:
            pair_keys.append((ca, cb))

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pair_colors = {pk: prop_cycle[i % len(prop_cycle)] for i, pk in enumerate(pair_keys)}

    # ------------------------------------------------------------------
    # Figura
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    D_ref = dist_by_layer[layer_idx]
    C_ref = cross_by_layer[layer_idx]
    tick_labels = labels

    def _heatmap(ax, data, title, cmap, fmt=".2f"):
        np_data = data.detach().cpu().numpy()
        im = ax.imshow(np_data, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(tick_labels)))
        ax.set_yticklabels(tick_labels, fontsize=6)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _class_heatmap(ax, means, title, cmap, fmt=".2f"):
        M = _pair_matrix(means, unique_cats).detach().cpu().numpy()
        im = ax.imshow(M, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(unique_cats)))
        ax.set_xticklabels(unique_cats, fontsize=8)
        ax.set_yticks(range(len(unique_cats)))
        ax.set_yticklabels(unique_cats, fontsize=8)
        for i in range(len(unique_cats)):
            for j in range(len(unique_cats)):
                ax.text(j, i, f"{M[i, j]:{fmt}}", ha="center", va="center",
                        fontsize=8, color="white" if M[i, j] < M.max() * 0.6 else "black")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _evolution_plot(ax, by_layer, ylabel, marker="o"):
        xs = all_layers
        for pk in pair_keys:
            ys = [by_layer[li].get(pk, by_layer[li].get((pk[1], pk[0]), float("nan")))
                  for li in xs]
            ax.plot(xs, ys, marker=marker, label=f"{pk[0]}-{pk[1]}",
                    color=pair_colors[pk], linewidth=1.4, markersize=4)
        ax.set_xlabel("Camada", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=7, loc="best")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Linha 0 — distância RKHS
    _heatmap(axes[0, 0], D_ref,
             f"D_RKHS completo · camada {layer_idx}", "viridis")
    _class_heatmap(axes[0, 1], pair_dist_by_layer[layer_idx],
                   "Média por classe", "viridis")
    _evolution_plot(axes[0, 2], pair_dist_by_layer,
                    "distância", marker="o")
    axes[0, 2].set_title("Distância RKHS média por camada", fontsize=9)

    # Linha 1 — cruzamentos de nível
    _heatmap(axes[1, 0], C_ref,
             f"Cruzamentos de nível · camada {layer_idx}", "YlOrRd")
    _class_heatmap(axes[1, 1], pair_cross_by_layer[layer_idx],
                   "Cruzamentos médios por classe", "YlOrRd", fmt=".1f")
    _evolution_plot(axes[1, 2], pair_cross_by_layer,
                    "# curvas", marker="s")
    axes[1, 2].set_title("Cruzamentos médios por camada", fontsize=9)

    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

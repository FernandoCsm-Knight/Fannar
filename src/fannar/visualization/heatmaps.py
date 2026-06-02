"""Heatmaps de matrizes de Gram e de distância."""

from __future__ import annotations

from typing import Any

import torch


def _np(t: torch.Tensor):
    return t.detach().cpu().numpy()


def _heatmap(M: torch.Tensor, title: str, cmap: str, labels: list[str] | None, ax: Any):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(_np(M), cmap=cmap, aspect="auto")
    ax.figure.colorbar(im, ax=ax)
    if labels is not None and len(labels) <= 40:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(title)
    return ax


def plot_gram_heatmap(K: torch.Tensor, labels: list[str] | None = None, ax: Any = None):
    return _heatmap(K, "Matriz de Gram", "viridis", labels, ax)


def plot_distance_heatmap(D: torch.Tensor, labels: list[str] | None = None, ax: Any = None):
    return _heatmap(D, "Matriz de distâncias", "magma", labels, ax)

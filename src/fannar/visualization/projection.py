"""Plots de projeção espectral 2D/3D.

Visualização não calcula geometria: recebe coordenadas/Gram e apenas desenha.
matplotlib é importado preguiçosamente.
"""

from __future__ import annotations

from typing import Any

import torch

from ..gram.eigenspace import embedding_fidelity, spectral_coordinates


def _np(t: torch.Tensor):
    return t.detach().cpu().numpy()


def plot_spectral_projection(
    K: torch.Tensor,
    dim: int = 2,
    labels: list[str] | None = None,
    color: torch.Tensor | None = None,
    ax: Any = None,
    title: str | None = None,
):
    """Projeta os objetos da Gram em 2D ou 3D e reporta a fidelidade no título."""
    import matplotlib.pyplot as plt

    coords = _np(spectral_coordinates(K, dim=dim))
    fid = embedding_fidelity(K, dim=dim)
    c = _np(color) if color is not None else None

    if dim == 2:
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=c, cmap="viridis", s=40)
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
    elif dim == 3:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=c, cmap="viridis", s=40)
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_zlabel("z3")
    else:
        raise ValueError("dim deve ser 2 ou 3.")

    if labels is not None:
        for i, name in enumerate(labels):
            pos = coords[i]
            ax.text(*pos, name, fontsize=7)
    if c is not None:
        ax.figure.colorbar(sc, ax=ax)
    ax.set_title(title or f"Projeção espectral {dim}D (fidelidade={fid:.2f})")
    return ax

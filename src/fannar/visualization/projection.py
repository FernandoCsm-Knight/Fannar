"""Plots de projeção espectral 2D/3D.

Visualização não calcula geometria: recebe coordenadas/Gram e apenas desenha.
matplotlib é importado preguiçosamente.
"""

from __future__ import annotations

from typing import Any

import torch

from ..gram.eigenspace import embedding_fidelity, kernel_pca, spectral_coordinates


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


def plot_rkhs_spectral_projection_2d(
    sample_kernels: dict[int, torch.Tensor],
    labels: list[str],
    categories: list[str] | None = None,
    layer_indices: list[int] | None = None,
    figsize: tuple[float, float] = (14, 8),
    suptitle: str = (
        "Geometria do espaço de representação: RKHS por camada via Moore-Aronszajn\n"
        r"$K_\mathrm{tipo} = K_W \odot K_A \odot K_G$ é PSD; "
        "sua decomposição espectral fornece a projeção dos exemplos."
    ),
) -> Any:
    """Grade 2×3 de projeções Kernel PCA 2D por camada.

    Cada painel corresponde a uma camada; cada ponto é uma amostra (frase/texto)
    posicionada pelas coordenadas espectrais √λ₁ v₁ e √λ₂ v₂ do kernel
    K_W ⊙ K_A ⊙ K_G centralizado (Kernel PCA).

    Parameters
    ----------
    sample_kernels : dict camada → K_tipos (N, N)
        Kernels entre N amostras por camada (saída de ``build_sample_kernel``).
    labels : list[str]
        Rótulo de cada amostra (ex.: ``["main", "animal", "food", "tech", ...]``).
    categories : list[str] | None
        Categoria de cada amostra para coloração; se None usa ``labels``.
    layer_indices : list[int] | None
        Seis índices de camada a exibir; se None seleciona 6 igualmente espaçados.
    figsize : tuple
        Tamanho da figura.
    suptitle : str
        Título principal da figura.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cats = categories if categories is not None else labels

    # seleciona 6 camadas
    all_layers = sorted(sample_kernels.keys())
    if layer_indices is None:
        n = len(all_layers)
        step = max(1, n // 6)
        layer_indices = all_layers[::step][:6]

    # mapa de cores por categoria
    unique_cats = sorted(set(cats))
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cat_to_color = {c: prop_cycle[i % len(prop_cycle)] for i, c in enumerate(unique_cats)}
    point_colors = [cat_to_color[c] for c in cats]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes_flat = axes.flatten()

    for ax_idx, layer_idx in enumerate(layer_indices[:6]):
        ax = axes_flat[ax_idx]
        K = sample_kernels[layer_idx]
        coords, _, explained = kernel_pca(K, n_components=2)
        xy = _np(coords)

        ax.scatter(xy[:, 0], xy[:, 1], s=90, c=point_colors, edgecolor="black", zorder=3)
        for i, lbl in enumerate(labels):
            ax.annotate(
                f"{i}:{lbl}",
                (xy[i, 0], xy[i, 1]),
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0),
            )

        ax.set_xlabel(r"$\sqrt{\lambda_1}\,v_1$")
        ax.set_ylabel(r"$\sqrt{\lambda_2}\,v_2$")
        ax.set_title(
            f"Camada {layer_idx} · Kernel PCA 2D · {100.0 * explained:.1f}% energia"
        )
        ax.grid(True, linestyle="--", alpha=0.4)

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o", color="w",
            markerfacecolor=cat_to_color[c], markersize=9,
            label=c,
        )
        for c in unique_cats
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(unique_cats),
        frameon=True,
        fontsize=9,
    )

    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig

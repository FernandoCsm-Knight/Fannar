"""Visualização da composição progressiva do espaço RKHS por produto tensorial.

Cada painel da grade 2×3 mostra os tokens de uma frase projetados em 2D via
MDS clássico sobre a matriz de distâncias acumulada até uma determinada camada.
O stress de Kruskal no título mede a fidelidade da projeção 2D.
"""

from __future__ import annotations

from typing import Any

import torch

from ..gram.eigenspace import classical_mds


def _robust_clip(coords: torch.Tensor, p: float = 1.0) -> torch.Tensor:
    """Recorte robusto: clampa coordenadas ao intervalo [p%, 100-p%] por eixo."""
    lo = torch.quantile(coords, p / 100.0, dim=0)
    hi = torch.quantile(coords, 1.0 - p / 100.0, dim=0)
    return coords.clamp(min=lo, max=hi)


def _needs_robust(coords: torch.Tensor, threshold: float = 3.0) -> bool:
    """Detecta outliers: True se alguma coord excede threshold × std da coluna."""
    std = coords.std(dim=0).clamp_min(1e-12)
    center = coords.mean(dim=0)
    return bool(((coords - center).abs() > threshold * std).any())


def plot_token_trajectories(
    cumulative_distances: list[torch.Tensor],
    token_labels: list[str],
    layer_indices: list[int] | None = None,
    n_components: int = 2,
    robust_threshold: float = 3.0,
    robust_clip_pct: float = 1.0,
    figsize: tuple[float, float] = (16, 9),
    suptitle: str = (
        "Composição progressiva do espaço de representação por produto tensorial de camadas\n"
        "Cada painel mostra os tokens no RKHS acumulado até a camada indicada. "
        "O produto entre camadas privilegia separações confirmadas ao longo da hierarquia."
    ),
    token_colors: list[Any] | None = None,
) -> Any:
    """Grade 2×3 de trajetórias de tokens via MDS sobre distâncias acumuladas.

    Cada painel corresponde a um acúmulo até uma camada específica.
    O primeiro painel mostra apenas a camada 0 (``H₀``); os demais mostram
    ``⊗ℓ≤k`` (produto tensorial até a camada k).

    Parameters
    ----------
    cumulative_distances : lista de (S, S) matrizes de distâncias acumuladas,
        uma por camada — saída de ``cumulative_token_distances``.
    token_labels         : rótulo de cada token (ex.: ``["The", "cat", ...]``).
    layer_indices        : índices originais das camadas correspondentes a cada
        entrada de ``cumulative_distances``.  Se ``None``, usa 0, 1, 2, ...
    n_components         : dimensão MDS (padrão 2).
    robust_threshold     : número de desvios padrão acima do qual se aplica
        recorte robusto de coordenadas (padrão 3.0).
    robust_clip_pct      : percentil de recorte quando robusto é ativado (padrão 1%).
    figsize              : tamanho da figura.
    suptitle             : título principal da figura.
    token_colors         : lista de cores para cada token; se ``None``,
        usa o ciclo de cores do matplotlib.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    n_panels = len(cumulative_distances)
    if layer_indices is None:
        layer_indices = list(range(n_panels))

    # seleciona 6 painéis igualmente espaçados
    if n_panels <= 6:
        panel_indices = list(range(n_panels))
    else:
        step = (n_panels - 1) / 5
        panel_indices = [round(i * step) for i in range(6)]

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if token_colors is None:
        token_colors = [prop_cycle[i % len(prop_cycle)] for i in range(len(token_labels))]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes_flat = axes.flatten()

    for ax_idx, panel_pos in enumerate(panel_indices):
        ax = axes_flat[ax_idx]
        D = cumulative_distances[panel_pos]
        layer_i = layer_indices[panel_pos]

        coords, stress = classical_mds(D, n_components=n_components)

        robust = _needs_robust(coords, threshold=robust_threshold)
        if robust:
            coords = _robust_clip(coords, p=robust_clip_pct)

        xy = coords.detach().cpu().numpy()

        ax.scatter(xy[:, 0], xy[:, 1], s=80,
                   c=token_colors[:len(token_labels)],
                   edgecolor="black", zorder=3)

        for i, lbl in enumerate(token_labels):
            ax.annotate(
                lbl, (xy[i, 0], xy[i, 1]),
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0),
            )

        label = "H₀" if panel_pos == 0 else f"⊗ℓ≤{layer_i}"
        robust_tag = " · recorte robusto" if robust else ""
        ax.set_title(f"{label} | stress={stress:.2f}{robust_tag}", fontsize=9)
        ax.set_xlabel("MDS 1", fontsize=8)
        ax.set_ylabel("MDS 2", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig

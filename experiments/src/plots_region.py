"""Figuras da correspondencia entre regioes de duas imagens (ver `src/theme.py`)."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import PowerNorm
from scipy.ndimage import zoom

from . import theme as T

T.use()

import matplotlib.pyplot as plt  # noqa: E402,F401  (tema antes do pyplot)

DISPLAY_GAMMA = 0.5
QUERY = T.PALETTE[1]
MATCH = T.PALETTE[2]


def _cell_center(cell, grid, size):
    r, c = cell
    h, w = grid
    return (c + 0.5) * size / w - 0.5, (r + 0.5) * size / h - 0.5


def _crop(image: np.ndarray, cell, grid, window: int) -> np.ndarray:
    size = image.shape[0]
    x, y = _cell_center(cell, grid, size)
    half = window // 2
    y0 = int(max(0, min(size - window, round(y) - half)))
    x0 = int(max(0, min(size - window, round(x) - half)))
    return image[y0 : y0 + window, x0 : x0 + window]


def plot_grid(images: np.ndarray, names: list[str], grid: int, path: str) -> None:
    """Imagens com a grade do bloco numerada, para escolher a posição de consulta."""
    cols = min(3, len(images))
    rows = int(np.ceil(len(images) / cols))
    fig, axes = T.grid(rows, cols, width="full", height=2.5 * rows)
    for k, ax in enumerate(axes.ravel()):
        if k >= len(images):
            ax.set_visible(False)
            continue
        ax.imshow(images[k], extent=(-0.5, grid - 0.5, grid - 0.5, -0.5))
        ax.set_xticks(np.arange(0, grid, 2))
        ax.set_yticks(np.arange(0, grid, 2))
        ax.set_xticks(np.arange(-0.5, grid, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid, 1), minor=True)
        ax.grid(which="minor", color=T.SURFACE, linewidth=0.3, alpha=0.6)
        ax.tick_params(labelsize=5.5, length=2)
        ax.set_title(names[k], fontsize=7.5)
    fig.tight_layout()
    T.save(fig, path)


def plot_matches(
    images: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    results: list[dict],
    window: int,
    path: str,
) -> None:
    """Por consulta: o recorte consultado e, em cada imagem da outra classe, onde ele casa."""
    rows = 2 * len(results)  # por consulta: mapas de proximidade e os recortes casados
    cols = 2 + max(len(r["matches"]) for r in results)
    fig, axes = T.grid(rows, cols, width="page", height=1.0 * rows + 0.4)
    for q, result in enumerate(results):
        r = 2 * q
        qi, qr, qc = result["query"]
        grid = result["grid"]
        image = images[qi]
        ax = axes[r, 0]
        ax.imshow(image)
        x, y = _cell_center((qr, qc), grid, image.shape[0])
        ax.plot(x, y, marker="*", ms=8, color=QUERY, markeredgecolor="white", markeredgewidth=0.7)
        T.bare(ax)
        ax.set_ylabel(f"{classes[labels[qi]]} {qi}\nlinha {qr}, coluna {qc}", fontsize=6.0,
                      rotation=0, ha="right", va="center")
        axes[r, 1].imshow(_crop(image, (qr, qc), grid, window))
        T.bare(axes[r, 1])
        axes[r, 1].set_xlabel("consulta", fontsize=5.8)
        axes[r + 1, 0].set_visible(False)
        axes[r + 1, 1].set_visible(False)

        for k, match in enumerate(result["matches"]):
            ax = axes[r, k + 2]
            target = images[match["target"]]
            size = target.shape[0]
            field = match["field"]
            near = field.max() - field
            m = zoom(near, size / near.shape[0], order=1)
            m = (m - m.min()) / max(m.max() - m.min(), 1e-12)
            ax.imshow(target.mean(-1), cmap="gray", vmin=0, vmax=255, alpha=0.5)
            ax.imshow(m, cmap=T.SEQ, norm=PowerNorm(DISPLAY_GAMMA, vmin=0, vmax=1), alpha=0.65)
            bx, by = _cell_center(match["best"], field.shape, size)
            ax.plot(bx, by, marker="*", ms=7, color=MATCH, markeredgecolor="white",
                    markeredgewidth=0.6)
            T.bare(ax)
            mark = " rec." if match["reciprocal"] else ""
            ax.set_xlabel(f"z {match['z']:+.1f}{mark}".replace(".", ","), fontsize=5.8)
            crop = axes[r + 1, k + 2]
            crop.imshow(_crop(target, match["best"], field.shape, window))
            T.bare(crop)
            crop.set_xlabel("casou aqui", fontsize=5.5)
        for k in range(len(result["matches"]) + 2, cols):
            axes[r, k].set_visible(False)
            axes[r + 1, k].set_visible(False)
    for col, title in enumerate(["imagem consultada", "recorte"]):
        axes[0, col].set_title(title, fontsize=7.0)
    axes[0, 2].set_title("proximidade à consulta em outras imagens (rec. = correspondência mútua)",
                         fontsize=7.0, loc="left")
    fig.tight_layout(pad=0.2, w_pad=0.15, h_pad=0.25)
    T.save(fig, path)

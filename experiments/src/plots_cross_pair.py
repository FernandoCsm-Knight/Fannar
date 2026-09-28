"""Figuras da Gram cruzada de duas imagens (ver `src/theme.py`)."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import PowerNorm
from scipy.ndimage import zoom

from . import theme as T

T.use()

import matplotlib.pyplot as plt  # noqa: E402,F401  (tema antes do pyplot)

DISPLAY_GAMMA = 0.5


def _upsample(m: np.ndarray, size: int) -> np.ndarray:
    return zoom(m, size / m.shape[0], order=1)


def _heat(ax, image: np.ndarray, field: np.ndarray, reverse: bool = False) -> None:
    size = image.shape[0]
    m = _upsample(field, size)
    if m.max() > m.min():
        m = (m - m.min()) / (m.max() - m.min())
    if reverse:
        m = 1.0 - m
    ax.imshow(image.mean(-1), cmap="gray", vmin=0, vmax=255, alpha=0.55)
    ax.imshow(m, cmap=T.SEQ, norm=PowerNorm(DISPLAY_GAMMA, vmin=0, vmax=1), alpha=0.72)
    T.bare(ax)


def _bands(ax, image: np.ndarray, p: np.ndarray, n_bands: int, ref=None) -> None:
    size = image.shape[0]
    m = _upsample(p, size)
    edges = np.unique(np.quantile(m, np.linspace(0, 1, n_bands + 1)[1:-1]))
    ax.imshow(image.mean(-1), cmap="gray", vmin=0, vmax=255, alpha=0.55)
    ax.imshow(np.digitize(m, edges), cmap=T.SEQ.resampled(len(edges) + 1).reversed(),
              vmin=-0.5, vmax=len(edges) + 0.5, alpha=0.72, interpolation="nearest")
    if len(edges):
        ax.contour(m, levels=edges, colors=T.INK, linewidths=0.4)
    if ref is not None:
        r, c = ref
        ax.plot((c + 0.5) * size / p.shape[1] - 0.5, (r + 0.5) * size / p.shape[0] - 0.5,
                marker="*", ms=7, color=T.PALETTE[1], markeredgecolor="white", markeredgewidth=0.6)
    T.bare(ax)


def plot_cross_pairs(
    images: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    classes: list[str],
    pairs: list[dict],
    block: str,
    n_bands: int,
    path: str,
) -> None:
    """Por par: as duas imagens, o que não tem correspondente em cada uma, e os níveis cruzados."""
    titles = ["imagem A", "imagem B", "sem correspondente em B", "sem correspondente em A",
              r"níveis de $s_0$ em A", r"os mesmos níveis em B"]
    fig, axes = T.grid(len(pairs), 6, width="page", height=1.15 * len(pairs) + 0.35)
    for row, pair in enumerate(pairs):
        a, b = images[pair["i"]], images[pair["j"]]
        for col, (img, index) in enumerate(((a, pair["i"]), (b, pair["j"]))):
            axes[row, col].imshow(img)
            T.bare(axes[row, col])
            axes[row, col].set_xlabel(
                f"{classes[labels[index]]} → {classes[preds[index]]}", fontsize=6.0)
        _heat(axes[row, 2], a, pair["no_counterpart_a"])
        _heat(axes[row, 3], b, pair["no_counterpart_b"])
        _bands(axes[row, 4], a, pair["p_a"], n_bands, pair["s0"])
        _bands(axes[row, 5], b, pair["p_b"], n_bands)
        axes[row, 0].set_ylabel(
            f"{pair['role']}\ngap {pair['gap']:.2f}".replace(".", ","),
            fontsize=6.5, rotation=0, ha="right", va="center")
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=7.0)
    fig.tight_layout(pad=0.2, w_pad=0.15, h_pad=0.2)
    T.save(fig, path)


def plot_cross_tests(summary: dict, blocks: list[str], roles: tuple[str, ...], path: str) -> None:
    """Gap mediano da margem de decisão por bloco: método, referências e pisos."""
    fig, axes = T.grid(1, 1, width="wide", height=2.6)
    ax = axes[0, 0]
    x = np.arange(len(blocks))
    ax.plot(x, [summary[b]["gap"] for b in blocks], lw=1.4, ms=3.6, **T.series("joint"))
    for key, rep, label in (("gap_gradcam", "grads", "Grad-CAM contrastivo"),
                            ("gap_other", "activations", "mapa contra C (especificidade)")):
        style = T.series(rep)
        style["label"] = label
        ax.plot(x, [summary[b][key] for b in blocks], lw=1.0, ms=3.0, **style)
    ax.plot(x, [summary[b]["gap_cross"] for b in blocks], lw=1.0, ms=3.0,
            **T.control("piso cruzado (rede não treinada)"))
    ax.plot(x, [summary[b]["gap_center"] for b in blocks], color=T.CONTROL, lw=0.9, ls="--",
            marker="s", ms=2.8, markerfacecolor=T.SURFACE, markeredgecolor=T.CONTROL, label="só o centro")
    ax.plot(x, [summary[b]["gap_random"] for b in blocks], color=T.CONTROL, lw=0.9,
            ls=(0, (1.0, 1.6)), marker="o", ms=3.0, markerfacecolor=T.SURFACE,
            markeredgecolor=T.CONTROL, label="ordem aleatória")
    ax.axhline(0.0, color=T.RULE, lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(blocks)
    ax.set_xlim(-0.35, len(blocks) - 0.65)
    ax.set_xlabel("bloco")
    ax.set_ylabel(r"gap da margem $\ell_{c_A}-\ell_{c_B}$ em A")
    ax.legend(fontsize=6.5, ncol=2)
    T.despine(ax)
    T.decimal(ax, "y", 2)
    fig.tight_layout()
    T.save(fig, path)

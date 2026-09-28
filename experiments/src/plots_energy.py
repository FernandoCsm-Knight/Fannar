"""Figuras da energia espectral por pixel, no formato do artigo.

Os mapas sao desenhados sobre a imagem em tons de cinza, com as curvas de nivel do
campo (quantis 50, 75 e 90%) por cima: sao os "mapas de calor e suas curvas de nivel"
do Comentario `rem:diagnostic`. Cada mapa e normalizado pelo proprio maximo, entao a
cor compara posicoes dentro de um painel e nao entre paineis.

Escala de cor nao linear: o campo tem cauda longa (CV entre pixels de 3 a 6 na conjunta),
entao em escala linear poucos pixels fixam o maximo e o resto vira branco. A cor usa
`PowerNorm(DISPLAY_GAMMA)`; com gamma < 1 a faixa baixa e esticada. E so visualizacao --
nenhum numero muda, e as curvas de nivel, por estarem em quantis, tambem nao. A legenda
da figura deve declarar o gamma. `DISPLAY_GAMMA = 1` volta a escala linear.

Identidade visual (paleta, tipografia, larguras, marcadores) vem de `src/theme.py`.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import PowerNorm
from scipy.ndimage import zoom

from . import theme as T

T.use()

import matplotlib.pyplot as plt  # noqa: E402,F401  (tema antes do pyplot)

GRADCAM = {"color": T.PALETTE[5], "marker": "X", "label": "Grad-CAM", "markeredgewidth": 0.0}
CROSS = T.control("controle cruzado (mapa da rede aleatória)", marker="s")
RANDOM = {**T.control("ordem aleatória", marker="o"), "linestyle": (0, (1.0, 1.6))}
METHODS = ("joint", "activations", "params", "grads")
DISPLAY_GAMMA = 0.5  # expoente do PowerNorm dos mapas; os scripts sobrescrevem por --display-gamma


def style(key: str) -> dict:
    if key == "gradcam":
        return dict(GRADCAM)
    if key == "cross":
        return dict(CROSS)
    if key == "random":
        return dict(RANDOM)
    return T.series(key)


def _upsample(m: np.ndarray, size: int = 32) -> np.ndarray:
    return zoom(m, size / m.shape[0], order=1)


def _overlay(ax, image: np.ndarray, field: np.ndarray) -> None:
    ax.imshow(image.mean(-1), cmap="gray", vmin=0, vmax=255, alpha=0.55)
    if np.isfinite(field).all() and field.max() > field.min():
        m = _upsample(field, image.shape[0])
        m = (m - m.min()) / (m.max() - m.min())
        ax.imshow(m, cmap=T.SEQ, norm=PowerNorm(DISPLAY_GAMMA, vmin=0, vmax=1), alpha=0.7)
        levels = np.unique(np.quantile(m, [0.5, 0.75, 0.9]))
        if len(levels) > 1:
            ax.contour(m, levels=levels, colors=T.INK, linewidths=[0.3, 0.45, 0.6][: len(levels)])
    else:
        ax.text(0.5, 0.5, "indefinido", transform=ax.transAxes, ha="center", va="center",
                fontsize=6.0, color=T.INK_FAINT)
    T.bare(ax)


def plot_field_grid(
    images: np.ndarray,
    fields: dict[str, np.ndarray],
    columns: list[str],
    titles: list[str],
    row_labels: list[str],
    path: str,
    masks: np.ndarray | None = None,
) -> None:
    """Uma linha por imagem; primeira coluna a imagem, depois um campo por coluna.

    Com `masks`, o contorno do animal (verdade externa) e desenhado sobre a imagem.
    """
    k = len(images)
    cols = 1 + len(columns)
    fig, axes = T.grid(k, cols, width="page", height=0.95 * k + 0.25)
    for i in range(k):
        ax = axes[i, 0]
        ax.imshow(images[i])
        if masks is not None:
            ax.contour(masks[i], levels=[0.5], colors=[T.PALETTE[1]], linewidths=0.8)
        T.bare(ax)
        ax.set_ylabel(row_labels[i], fontsize=6.5, rotation=0, ha="right", va="center")
        for j, key in enumerate(columns):
            _overlay(axes[i, j + 1], images[i], fields[key][i])
    for j, title in enumerate(["imagem", *titles]):
        axes[0, j].set_title(title, fontsize=7.5)
    fig.tight_layout(pad=0.2, w_pad=0.15, h_pad=0.15)
    T.save(fig, path)


def _block_axis(ax, blocks: list[str]) -> np.ndarray:
    x = np.arange(len(blocks))
    ax.set_xticks(x)
    ax.set_xticklabels(blocks)
    ax.set_xlim(-0.35, len(blocks) - 0.65)
    T.despine(ax)
    return x


def _grid_to_display(point: tuple[int, int], grid: tuple[int, int], size: int) -> tuple[float, float]:
    """(linha, coluna) na grade da analise -> (x, y) em pixels da imagem exibida."""
    (r, c), (h, w) = point, grid
    return (c + 0.5) * size / w - 0.5, (r + 0.5) * size / h - 0.5


def _bands(ax, image: np.ndarray, p: np.ndarray, n_bands: int, ref: tuple[int, int] | None,
           marker_color) -> None:
    """Faixas de nivel Γ_{r,δ}: raios nos quantis de p, faixa mais proxima = mais escura."""
    size = image.shape[0]
    m = _upsample(p, size)
    edges = np.unique(np.quantile(m, np.linspace(0, 1, n_bands + 1)[1:-1]))
    band = np.digitize(m, edges)
    cmap = T.SEQ.resampled(len(edges) + 1).reversed()
    ax.imshow(image.mean(-1), cmap="gray", vmin=0, vmax=255, alpha=0.55)
    ax.imshow(band, cmap=cmap, vmin=-0.5, vmax=len(edges) + 0.5, alpha=0.72, interpolation="nearest")
    if len(edges):
        ax.contour(m, levels=edges, colors=T.INK, linewidths=0.4)
    if ref is not None:
        x, y = _grid_to_display(ref, p.shape, size)
        ax.plot(x, y, marker="*", ms=7, color=marker_color, markeredgecolor="white", markeredgewidth=0.6)
    T.bare(ax)


def plot_level_set_detail(
    images: np.ndarray,
    readouts: list[dict],
    row_labels: list[str],
    path: str,
    n_bands: int = 6,
) -> None:
    """Um bloco: energia, p_{s0} contínua com curvas, faixas Γ_{r,δ}(s0), N_k(s0), faixas de s1."""
    k = len(images)
    titles = ["imagem", r"$\hat E_\parallel$", r"$p_{s_0}$ e curvas de nível",
              r"faixas $\Gamma_{r,\delta}(s_0)$", r"$\mathcal{N}_k(s_0)$", r"faixas $\Gamma_{r,\delta}(s_1)$"]
    fig, axes = T.grid(k, len(titles), width="full", height=1.2 * k + 0.25)
    for i in range(k):
        img, r = images[i], readouts[i]
        size = img.shape[0]
        axes[i, 0].imshow(img)
        T.bare(axes[i, 0])
        axes[i, 0].set_ylabel(row_labels[i], fontsize=6.5, rotation=0, ha="right", va="center")

        _overlay(axes[i, 1], img, r["energy"])
        for ref, color in ((r["s0"], T.PALETTE[1]), (r["s1"], T.PALETTE[2])):
            x, y = _grid_to_display(ref, r["energy"].shape, size)
            axes[i, 1].plot(x, y, marker="*", ms=6, color=color, markeredgecolor="white", markeredgewidth=0.5)

        ax = axes[i, 2]
        m = _upsample(r["p0"], size)
        # a potencia vai sobre a proximidade max(p) - p: com gamma < 1 ela realca os pixels
        # proximos de s0 (escuros), em vez de clarea-los como faria sobre a propria distancia
        near = m.max() - m
        ax.imshow(near, cmap=T.SEQ, norm=PowerNorm(DISPLAY_GAMMA, vmin=0, vmax=max(near.max(), 1e-12)))
        levels = np.unique(np.quantile(m, np.linspace(0, 1, n_bands + 1)[1:-1]))
        if len(levels):
            ax.contour(m, levels=levels, colors=T.INK, linewidths=0.45)
        x, y = _grid_to_display(r["s0"], r["p0"].shape, size)
        ax.plot(x, y, marker="*", ms=6, color=T.PALETTE[1], markeredgecolor="white", markeredgewidth=0.5)
        T.bare(ax)

        _bands(axes[i, 3], img, r["p0"], n_bands, r["s0"], T.PALETTE[1])

        ax = axes[i, 4]
        ax.imshow(img.mean(-1), cmap="gray", vmin=0, vmax=255, alpha=0.55)
        mask = zoom(r["knn"].astype(float), size / r["knn"].shape[0], order=0)
        ax.imshow(np.ma.masked_where(mask < 0.5, mask), cmap=T.SEQ, vmin=0, vmax=1, alpha=0.85,
                  interpolation="nearest")
        ax.plot(x, y, marker="*", ms=6, color=T.PALETTE[1], markeredgecolor="white", markeredgewidth=0.5)
        T.bare(ax)

        _bands(axes[i, 5], img, r["p1"], n_bands, r["s1"], T.PALETTE[2])
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=7.5)
    fig.tight_layout(pad=0.2, w_pad=0.15, h_pad=0.15)
    T.save(fig, path)


def plot_pets_metrics(stats: dict, gap: dict, level: dict, level_u: dict, blocks: list[str],
                      path: str) -> None:
    """(a) energia no animal acima do acaso, (b) gap de delecao, (c) N_k(s0) sobre o animal."""
    fig, axes = T.grid(1, 3, width="full", height=2.5)
    styles = {"joint": T.series("joint"), "activations": T.series("activations"),
              "gradcam": dict(GRADCAM), "cross": dict(CROSS),
              "center": {**T.control("só o centro", marker="D"), "linestyle": (0, (5, 1.5))},
              "random": dict(RANDOM)}
    x = _block_axis(axes[0, 0], blocks)
    for key, style in styles.items():
        if key == "random":
            continue
        axes[0, 0].plot(x, [stats[key][b]["lift"] for b in blocks], lw=1.2, ms=3.0, **style)
    axes[0, 0].axhline(0.0, color=T.RULE, lw=0.6)
    axes[0, 0].set_ylabel("energia no animal − fração do animal")
    axes[0, 0].set_title("o campo cai no animal?", fontsize=8.0)
    T.decimal(axes[0, 0], "y", 2)

    x = _block_axis(axes[0, 1], blocks)
    for key, style in styles.items():
        axes[0, 1].plot(x, [gap[key][b] for b in blocks], lw=1.2, ms=3.0, **style)
    axes[0, 1].axhline(0.0, color=T.RULE, lw=0.6)
    axes[0, 1].set_ylabel("gap LeRF − MoRF da margem")
    axes[0, 1].set_title("o campo aponta o que a rede usa?", fontsize=8.0)
    T.decimal(axes[0, 1], "y", 2)

    x = _block_axis(axes[0, 2], blocks)
    axes[0, 2].plot(x, [level[b]["knn_on_animal"] for b in blocks], lw=1.2, ms=3.0,
                    **{**T.series("joint"), "label": "rede treinada"})
    axes[0, 2].plot(x, [level_u[b]["knn_on_animal"] for b in blocks], lw=1.0, ms=3.0,
                    **T.control("rede aleatória"))
    axes[0, 2].plot(x, [level[b]["chance"] for b in blocks], color=T.RULE, lw=0.9, label="acaso")
    axes[0, 2].set_ylabel(r"fração de $\mathcal{N}_k(s_0)$ no animal")
    axes[0, 2].set_title("a região equivalente a $s_0$ é o animal?", fontsize=8.0)
    axes[0, 2].legend(fontsize=6.5)
    T.decimal(axes[0, 2], "y", 1)
    for ax in axes[0]:
        ax.set_xlabel("bloco")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=6, fontsize=6.6,
               bbox_to_anchor=(0.5, -0.1))
    T.panel_tags(axes)
    fig.tight_layout()
    T.save(fig, path)



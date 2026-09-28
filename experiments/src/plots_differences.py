"""Figuras de "por que esta amostra teve outra classificacao" (ver `src/theme.py`)."""

from __future__ import annotations

import numpy as np

from . import theme as T

T.use()

import matplotlib.pyplot as plt  # noqa: E402,F401  (tema antes do pyplot)

STYLE = {
    "method": T.series("joint"),
    "absdiff": T.series("activations"),
    "gradxdelta": T.series("rsa_euclid"),
    "random": {**T.control("ordem aleatória", marker="o"), "linestyle": (0, (1.0, 1.6))},
    "floor": T.control("método na rede não treinada"),
}


def _area(y, x):
    y, x = np.asarray(y), np.asarray(x)
    return float(((y[1:] + y[:-1]) * 0.5 * np.diff(x)).sum())


def plot_tabular_differences(runs: list[dict], names: list[str], blocks: list[str], block: str,
                             labels: dict, path: str) -> None:
    """(a) fração de pares que passam à classe de j ao longo das trocas, (b) área por bloco."""
    grid = np.asarray(runs[0]["grid"])
    per = {}
    for n in names:
        rs = [r for r in runs if r["dataset"] == n]
        per[n] = {k: np.mean([r["curves"][k] for r in rs], 0) for k in rs[0]["curves"]}
    fig, axes = T.grid(1, 2, width="full", height=2.6)
    ax = axes[0, 0]
    for key in (f"method@{block}", "absdiff", "gradxdelta", "random", f"floor@{block}"):
        kind = key.split("@")[0]
        curves = np.array([per[n][key] for n in names])
        m, s = curves.mean(0), curves.std(0)
        style = {**STYLE[kind], "label": labels[kind]}
        style.pop("marker", None)
        ax.fill_between(grid, m - s, m + s, color=style["color"], alpha=0.10, linewidth=0)
        ax.plot(grid, m, lw=1.4, **style)
    ax.set_xlabel("fração dos atributos de j trocados em i")
    ax.set_ylabel("fração dos pares na classe de j")
    ax.set_title(f"trocas guiadas por cada ranking (método em {block})", fontsize=8.0)
    ax.legend(fontsize=6.5, loc="lower right")
    T.despine(ax)
    T.decimal(ax, "both", 1)

    ax = axes[0, 1]
    x = np.arange(len(blocks))
    for kind in ("method", "floor"):
        vals = np.array([[_area(per[n][f"{kind}@{b}"], grid) for b in blocks] for n in names])
        m, s = vals.mean(0), vals.std(0)
        ax.fill_between(x, m - s, m + s, color=STYLE[kind]["color"], alpha=0.12, linewidth=0)
        ax.plot(x, m, lw=1.4, ms=3.2, **{**STYLE[kind], "label": labels[kind]})
    for ref in ("absdiff", "gradxdelta", "random"):
        v = np.mean([_area(per[n][ref], grid) for n in names])
        style = {k: v2 for k, v2 in STYLE[ref].items() if k in ("color", "linestyle")}
        ax.axhline(v, lw=1.0, **style, label=labels[ref])
    ax.set_xticks(x)
    ax.set_xticklabels(blocks)
    ax.set_xlabel("bloco da Gram usada pelo método")
    ax.set_ylabel("área sob a curva (maior = melhor)")
    T.despine(ax)
    T.decimal(ax, "y", 2)
    T.panel_tags(axes)
    fig.tight_layout()
    T.save(fig, path)

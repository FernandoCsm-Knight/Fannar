"""Figura do estudo de caso da arvore de decisao (P6): metodo × Shapley × arvore (ver `src/theme.py`)."""

from __future__ import annotations

import numpy as np
import seaborn as sns
from scipy.stats import spearmanr

from . import theme as T

T.use()

import matplotlib.pyplot as plt  # noqa: E402,F401  (tema antes do pyplot)

_EXTRA = sns.color_palette("colorblind", 10)  # as duas fatias fora da ordem canonica de `theme`
STYLE = {
    "method": T.series("joint"),
    "shap_baseline": T.series("params"),
    "shap": T.series("grads"),
    "lime": T.series("joint_root"),
    "ig": {"color": _EXTRA[9], "marker": "P", "markeredgewidth": 0.0},
    "dice": {"color": _EXTRA[8], "marker": "*", "markeredgewidth": 0.0},
    "absdiff": T.series("activations"),
    "gradxdelta": T.series("rsa_euclid"),
    "random": {**T.control("aleatória", marker="o"), "linestyle": (0, (1.0, 1.6))},
    "floor": T.control("método na rede não treinada"),
}
def _present(runs: list[dict], kinds) -> list[str]:
    """So as series que a execucao tem (o DiCE e opcional)."""
    return [k for k in kinds if k in ("method", "floor") or k in runs[0]["auc"]]


def _area(y, x):
    y, x = np.asarray(y), np.asarray(x)
    return float(((y[1:] + y[:-1]) * 0.5 * np.diff(x)).sum())


def _key(kind: str, block: str) -> str:
    return f"{kind}@{block}" if kind in ("method", "floor") else kind


def _norm(v):
    v = np.maximum(np.asarray(v, dtype=float), 0)
    return v / max(v.sum(), 1e-12)


TREE_STYLE = {
    "method": T.series("joint"),
    "shapley": T.series("params"),
    "absdiff": T.series("activations"),
    "random": {**T.control("aleatória", marker="o"), "linestyle": (0, (1.0, 1.6))},
    "floor": T.control("método na árvore de rótulos embaralhados"),
    "caminho": {**T.series("grads"), "linestyle": (0, (4, 1.5))},
    "folga": {**T.series("rsa_euclid"), "linestyle": (0, (4, 1.5))},
    "classes": {**T.series("joint_root"), "linestyle": (0, (4, 1.5))},
}


def plot_tree_case(runs: list[dict], labels: dict, path: str) -> None:
    """(a) trocas na propria arvore, (b) datacao contra a profundidade do corte,
    (c) posicao do atributo do corte, (d) importancia global contra o Gini."""
    grid = np.asarray(runs[0]["grid"])
    pairs = [p for r in runs for p in r["pairs"]]
    fig, axes = T.grid(2, 2, width="full", height=5.4)

    ax = axes[0, 0]
    for kind in ("method", "shapley", "absdiff", "random", "floor"):
        style = {**TREE_STYLE[kind], "label": labels[kind]}
        style.pop("marker", None)
        ax.plot(grid, np.mean([r["curves"][kind] for r in runs], 0), lw=1.4, **style)
    ax.set_xlabel("fração dos atributos de j trocados em i")
    ax.set_ylabel("fração dos pares na classe de j (árvore)")
    ax.legend(fontsize=6.0, loc="lower right")
    T.despine(ax)
    T.decimal(ax, "both", 1)

    ax = axes[0, 1]
    diff = np.array([p["first_top_depth"] - (p["cut_depth"] + 1) for p in pairs if p["first_top_depth"] is not None])
    lo, hi = max(diff.min(), -8), min(diff.max(), 8)
    bins = np.arange(lo - 0.5, hi + 1.5)
    ax.hist(np.clip(diff, lo, hi), bins=bins, color=TREE_STYLE["method"]["color"], rwidth=0.8,
            weights=np.full(len(diff), 1 / len(pairs)))
    ax.set_xlabel("atraso da datação (profundidades)")
    ax.set_ylabel("fração dos pares")
    T.despine(ax)
    T.decimal(ax, "y", 1)

    ax = axes[1, 0]
    kinds = ("method", "caminho", "folga", "classes", "shapley", "absdiff", "random", "floor")
    y = np.arange(len(kinds))[::-1]
    for yi, k in zip(y, kinds):
        v = np.array([np.mean([p["positions"][k] == 0 for p in r["pairs"]]) for r in runs])
        ax.plot([v.mean()], [yi], linestyle="none", ms=5,
                **{kk: vv for kk, vv in TREE_STYLE[k].items() if kk not in ("linestyle", "label")})
        ax.plot([v.mean() - v.std(), v.mean() + v.std()], [yi, yi], color=TREE_STYLE[k]["color"], lw=1.0)
    chance = np.mean([1 / p["n_differ"] for p in pairs])
    ax.axvline(chance, color=T.CONTROL, lw=0.9, linestyle=(0, (3.5, 2.5)))
    ax.text(chance, y[0] + 0.55, " acaso", color=T.INK_FAINT, fontsize=6.5, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([labels[k] for k in kinds], fontsize=6.5)
    ax.set_ylim(y[-1] - 0.6, y[0] + 1.1)
    ax.set_xlabel("pares com o atributo do corte em 1º")
    T.despine(ax)
    T.decimal(ax, "x", 1)

    ax = axes[1, 1]
    gini = np.array([_norm(r["global"]["gini"]) for r in runs]).mean(0)
    order = np.argsort(-gini)
    xs = np.arange(len(order))
    ax.bar(xs, gini[order], color="0.85", width=0.75, label="Gini da árvore (referência)")
    for kind, style, name in (("method", TREE_STYLE["method"], "método (parcela média)"),
                              ("tree_shap", TREE_STYLE["shapley"], "SHAP de árvore")):
        vals = np.array([_norm(r["global"][kind]) for r in runs])
        rho = np.mean([spearmanr(r["global"][kind], r["global"]["gini"])[0] for r in runs])
        ax.plot(xs, vals.mean(0)[order], linestyle="none", ms=4.0,
                **{**style, "label": f"{name}: {rho:.2f}".replace(".", ",")})
    from src.case import DATASET  # noqa: PLC0415
    from src.data_tabular import feature_names  # noqa: PLC0415
    feats = feature_names(DATASET)
    ax.set_xticks(xs)
    ax.set_xticklabels([feats[f] for f in order], rotation=60, ha="right", fontsize=6.0)
    ax.set_ylabel("parcela da importância")
    ax.legend(fontsize=5.8, loc="upper right")
    T.despine(ax)
    T.decimal(ax, "y", 2)

    T.panel_tags(axes)
    fig.tight_layout()
    T.save(fig, path)

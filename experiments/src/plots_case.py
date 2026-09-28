"""Figura do estudo de caso na `letter`: metodo × SHAP × arvore (ver `src/theme.py`)."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from . import theme as T

T.use()

import matplotlib.pyplot as plt  # noqa: E402,F401  (tema antes do pyplot)

STYLE = {
    "method": T.series("joint"),
    "shap_baseline": T.series("params"),
    "shap": T.series("grads"),
    "lime": T.series("joint_root"),
    "absdiff": T.series("activations"),
    "gradxdelta": T.series("rsa_euclid"),
    "random": {**T.control("aleatória", marker="o"), "linestyle": (0, (1.0, 1.6))},
    "floor": T.control("método na rede não treinada"),
}
CURVES = ("method", "shap_baseline", "shap", "lime", "absdiff", "gradxdelta", "random", "floor")


def _area(y, x):
    y, x = np.asarray(y), np.asarray(x)
    return float(((y[1:] + y[:-1]) * 0.5 * np.diff(x)).sum())


def _key(kind: str, block: str) -> str:
    return f"{kind}@{block}" if kind in ("method", "floor") else kind


def _norm(v):
    v = np.maximum(np.asarray(v, dtype=float), 0)
    return v / max(v.sum(), 1e-12)


def plot_case_study(runs: list[dict], block: str, labels: dict, global_labels: dict,
                    features: list[str], path: str) -> None:
    """(a) curvas de troca, (b) AUC por bloco, (c) atributo da arvore no topo do ranking,
    (d) importancia global contra a permutacao no MLP."""
    grid = np.asarray(runs[0]["grid"])
    blocks = runs[0]["blocks"]
    fig, axes = T.grid(2, 2, width="full", height=5.4)

    ax = axes[0, 0]
    for kind in CURVES:
        curves = np.array([r["curves"][_key(kind, block)] for r in runs])
        style = {**STYLE[kind], "label": labels[kind]}
        style.pop("marker", None)
        ax.plot(grid, curves.mean(0), lw=1.4, **style)
    ax.set_xlabel("fração dos atributos de j trocados em i")
    ax.set_ylabel("fração dos pares na classe de j")
    ax.legend(fontsize=6.0, loc="lower right")
    T.despine(ax)
    T.decimal(ax, "both", 1)

    ax = axes[0, 1]
    x = np.arange(len(blocks))
    for kind in ("method", "floor"):
        vals = np.array([[r["auc"][f"{kind}@{b}"] for b in blocks] for r in runs])
        m, s = vals.mean(0), vals.std(0)
        ax.fill_between(x, m - s, m + s, color=STYLE[kind]["color"], alpha=0.12, linewidth=0)
        ax.plot(x, m, lw=1.4, ms=3.2, **{**STYLE[kind], "label": labels[kind]})
    for ref in ("shap_baseline", "shap", "lime", "absdiff", "gradxdelta", "random"):
        v = np.mean([r["auc"][ref] for r in runs])
        style = {k: v2 for k, v2 in STYLE[ref].items() if k in ("color", "linestyle")}
        ax.axhline(v, lw=1.0, **style)  # mesmas series e estilos de (a), cuja legenda vale aqui
    ax.set_xticks(x)
    ax.set_xticklabels(blocks)
    ax.set_xlabel("bloco da Gram usada pelo método")
    ax.set_ylabel("área sob a curva (maior = melhor)")
    T.despine(ax)
    T.decimal(ax, "y", 2)

    # (c) o atributo em que a arvore substituta separa i de j esta entre os 3 primeiros?
    ax = axes[1, 0]
    kinds = ("method", "shap_baseline", "shap", "lime", "absdiff", "gradxdelta", "random", "floor")
    hits, chance = {k: [] for k in kinds}, []
    for r in runs:
        per, ch = {k: [] for k in kinds}, []
        for p in r["pairs"]:
            t = p["trees"]["surrogate"]
            if not t["reproduces"] or t["feature"] is None:
                continue
            for k in kinds:
                order = np.asarray(p["orders"][_key(k, block)])
                per[k].append(int(np.flatnonzero(order == t["feature"])[0]) < 3)
            ch.append(min(3, p["n_differ"]) / p["n_differ"])
        for k in kinds:
            hits[k].append(np.mean(per[k]))
        chance.append(np.mean(ch))
    y = np.arange(len(kinds))[::-1]
    for yi, k in zip(y, kinds):
        v = np.asarray(hits[k])
        ax.plot([v.mean()], [yi], linestyle="none", ms=5, **{kk: vv for kk, vv in STYLE[k].items()
                                                              if kk not in ("linestyle", "label")})
        ax.plot([v.mean() - v.std(), v.mean() + v.std()], [yi, yi], color=STYLE[k]["color"], lw=1.0)
    ax.axvline(np.mean(chance), color=T.CONTROL, lw=0.9, linestyle=(0, (3.5, 2.5)))
    ax.text(np.mean(chance), y[0] + 0.55, " acaso", color=T.INK_FAINT, fontsize=6.5, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([labels[k] for k in kinds], fontsize=6.5)
    ax.set_ylim(y[-1] - 0.6, y[0] + 1.1)
    ax.set_xlabel("pares com o atributo da árvore entre os 3 primeiros")
    T.despine(ax)
    T.decimal(ax, "x", 1)

    # (d) importancia global normalizada, na ordem da permutacao no MLP
    ax = axes[1, 1]
    perm = np.array([_norm(r["global"]["permutation"]) for r in runs]).mean(0)
    order = np.argsort(-perm)
    xs = np.arange(len(order))
    ax.bar(xs, perm[order], color="0.85", width=0.75, label=global_labels["permutation"])
    for kind, style in (("method", STYLE["method"]), ("shap", STYLE["shap"]), ("lime", STYLE["lime"]),
                        ("surrogate", {"color": T.INK, "marker": "x", "markeredgewidth": 0.9})):
        vals = np.array([_norm(r["global"][kind][block] if kind == "method" else r["global"][kind]) for r in runs])
        rho = np.mean([spearmanr(v, p)[0] for v, p in zip(vals, (r["global"]["permutation"] for r in runs))])
        ax.plot(xs, vals.mean(0)[order], linestyle="none", ms=4.0,
                **{**style, "label": f"{global_labels[kind]}, ρ = {rho:.2f}".replace(".", ",")})
    ax.set_xticks(xs)
    ax.set_xticklabels([features[f] for f in order], rotation=60, ha="right", fontsize=6.0)
    ax.set_ylabel("parcela da importância")
    ax.legend(fontsize=5.8, loc="upper right")
    T.despine(ax)
    T.decimal(ax, "y", 2)

    T.panel_tags(axes)
    fig.tight_layout()
    T.save(fig, path)


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
                **{**style, "label": f"{name}, ρ = {rho:.2f}".replace(".", ",")})
    from letter_case_study import DATASET  # noqa: PLC0415
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

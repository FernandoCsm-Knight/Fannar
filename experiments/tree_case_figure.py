"""Figura lado a lado: o caminho de decisao da arvore e a leitura do metodo sobre a mesma arvore.

Para cada par (i, j), a esquerda mostra os caminhos de i e de j na arvore de decisao da
`letter` (semente 0): o trecho comum, o no em que eles se separam e os ramos ate as folhas. A
direita, alinhada pela profundidade, mostra a parcela de cada atributo no Δd do metodo em cada
profundidade ℓ (fontes caminho, folga e classes; `tree_case_study.py`). A arvore se explica
pelo proprio caminho; a figura mostra o metodo recuperando essa explicacao -- o atributo do
corte passa a dominar exatamente na profundidade em que os caminhos se separam.

A arvore e reajustada com os mesmos dados e a mesma semente de `tree_case_study.py`, e o script
confere que o no de separacao coincide com o gravado em `outputs/tree_case/runs/seed0.json`.

Uso:
    python tree_case_figure.py --pairs 41 91
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle
from sklearn.tree import DecisionTreeClassifier

from letter_case_study import DATASET, split_node
from src import theme as T
from src.data_tabular import class_names, feature_names, load_tabular, split

T.use()

import matplotlib.pyplot as plt  # noqa: E402

AFTER = 3        # nos mostrados em cada ramo depois da separacao
MAX_FEATURES = 7


def path_nodes(tree, x) -> list[int]:
    return list(tree.decision_path(x[None].astype(np.float32)).indices)


def fmt(v: float) -> str:
    return f"{v:.1f}".replace(".", ",")


def node_label(tree, n: int, feats) -> str:
    t = tree.tree_
    if t.children_left[n] == -1:
        return ""
    return f"{feats[t.feature[n]]} ≤ {fmt(t.threshold[n])}"


def draw_tree(ax, tree, p, feats, classes, colors) -> dict:
    """Caminhos de i e j: trecho comum no centro, ramos a esquerda (i) e a direita (j)."""
    xi, xj = np.asarray(p["raw_i"]), np.asarray(p["raw_j"])
    pi, pj = path_nodes(tree, xi), path_nodes(tree, xj)
    cut = split_node(tree, xi, xj)
    c = pi.index(cut)                      # profundidade do no de separacao
    assert cut == int(np.intersect1d(pi, pj).max())
    leaf_y = -(c + 1 + AFTER + 1.2)
    box = dict(boxstyle="round,pad=0.28", linewidth=0.6)

    def put(x, y, text, edge, face=T.SURFACE, lw=0.6, weight="normal", color=T.INK):
        ax.text(x, y, text, ha="center", va="center", fontsize=6.2, color=color, fontweight=weight,
                bbox={**box, "edgecolor": edge, "facecolor": face, "linewidth": lw}, zorder=3)

    for d, n in enumerate(pi[: c + 1]):
        if d:
            ax.plot([0, 0], [-(d - 1), -d], color=T.RULE, lw=0.9, zorder=1)
        if n == cut:
            f = tree.tree_.feature[n]
            text = f"{node_label(tree, n, feats)}\ni = {xi[f]:.0f}   j = {xj[f]:.0f}"
            put(0, -d, text, T.INK, lw=1.3, weight="bold")
        else:
            put(0, -d, node_label(tree, n, feats), T.RULE)
    for side, path, x, color, who, pred in ((-1, pi, -0.62, colors[0], "i", p["pred_i"]),
                                           (1, pj, 0.62, colors[1], "j", p["pred_j"])):
        branch = path[c + 1:]
        shown = branch[:AFTER]
        prev = (0, -c)
        for k, n in enumerate(shown):
            y = -(c + 1 + k)
            ax.plot([prev[0], x], [prev[1], y], color=color, lw=1.1, zorder=1)
            leaf = tree.tree_.children_left[n] == -1
            if leaf:
                put(x, y, f"folha: {classes[pred]}", color, face=color, color=T.SURFACE, weight="bold")
            else:
                put(x, y, node_label(tree, n, feats), color)
            prev = (x, y)
            if leaf:
                break
        else:
            rest = len(branch) - len(shown)
            ax.plot([x, x], [prev[1], leaf_y], color=color, lw=1.1, linestyle=(0, (1, 1.5)), zorder=1)
            ax.text(x + 0.05 * side, (prev[1] + leaf_y) / 2, f"+{rest - 1} nós", fontsize=5.6,
                    color=T.INK_FAINT, ha="left" if side > 0 else "right", va="center")
            put(x, leaf_y, f"folha: {classes[pred]}", color, face=color, color=T.SURFACE, weight="bold")
    ax.text(-0.62, 0.55, f"i: {classes[p['pred_i']]}", color=colors[0], fontsize=7.5, ha="center",
            fontweight="bold")
    ax.text(0.62, 0.55, f"j: {classes[p['pred_j']]}", color=colors[1], fontsize=7.5, ha="center",
            fontweight="bold")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(leaf_y - 0.7, 0.9)
    ax.axis("off")
    return {"cut_depth": c, "leaf_y": leaf_y}


def draw_method(ax, p, feats, geom) -> None:
    """Parcela de cada atributo no Δd, por profundidade, com a mesma escala vertical da arvore."""
    c, leaf_y = geom["cut_depth"], geom["leaf_y"]
    full = np.asarray(p["method_full"])
    differ = np.flatnonzero(np.abs(np.asarray(p["raw_i"]) - np.asarray(p["raw_j"])) > 1e-9)
    cols = [f for f in np.argsort(-full) if f in differ][:MAX_FEATURES]
    rows = [(-ell, p["method_by_depth"][str(ell)]) for ell in range(1, c + AFTER + 1)]
    rows.append((leaf_y, p["method_full"]))
    cmap = T.SEQ
    for y, scores in rows:
        v = np.maximum(np.asarray(scores, dtype=float), 0)
        share = v / v.sum() if v.sum() > 0 else v
        for k, f in enumerate(cols):
            ax.add_patch(Rectangle((k - 0.5, y - 0.45), 1, 0.9, facecolor=cmap(0.08 + 0.92 * share[f]),
                                   edgecolor=T.SURFACE, linewidth=0.8))
            if share[f] >= 0.2:
                ax.text(k, y, f"{share[f]:.0%}".replace("%", ""), ha="center", va="center", fontsize=5.4,
                        color=T.SURFACE if share[f] > 0.55 else T.INK)
    k_cut = cols.index(p["cut_feature"]) if p["cut_feature"] in cols else None
    sep = -(c + 1)
    ax.add_patch(Rectangle((-0.5, sep - 0.5), len(cols), 1.0, fill=False, edgecolor=T.INK, linewidth=1.2,
                           linestyle=(0, (3, 1.5))))
    ax.text(len(cols) - 0.35, sep, "caminhos\nse separam", fontsize=5.6, va="center", ha="left",
            color=T.INK_SOFT)
    ax.text(len(cols) - 0.35, leaf_y, "árvore\ninteira", fontsize=5.6, va="center", ha="left",
            color=T.INK_SOFT)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([feats[f] for f in cols], rotation=55, ha="left", fontsize=6.2)
    ax.xaxis.tick_top()
    if k_cut is not None:
        lab = ax.get_xticklabels()[k_cut]
        lab.set_fontweight("bold")
    ys = [y for y, _ in rows]
    ax.set_yticks(ys)
    ax.set_yticklabels([f"ℓ = {-y}" for y in ys[:-1]] + ["máx."], fontsize=6.0)
    ax.set_xlim(-0.5, len(cols) + 1.6)
    ax.set_ylim(leaf_y - 0.7, 0.9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=int, nargs="+", default=[41, 91])
    ap.add_argument("--run", default="outputs/tree_case/runs/seed0.json")
    ap.add_argument("--min-leaf", type=int, default=3)
    ap.add_argument("--out", default="outputs/tree_case/tree_vs_method.pdf")
    args = ap.parse_args()

    run = json.loads(Path(args.run).read_text())
    seed = run["seed"]
    X, y = load_tabular(DATASET)
    tr, _, _ = split(y, seed)
    tree = DecisionTreeClassifier(min_samples_leaf=args.min_leaf, random_state=seed).fit(X[tr], y[tr])
    feats, classes = feature_names(DATASET), class_names(DATASET)
    colors = (T.PALETTE[0], T.PALETTE[1])

    n = len(args.pairs)
    fig, axes = T.grid(n, 2, width="full", height=3.3 * n, gridspec_kw={"width_ratios": [1.15, 1.0]})
    for r, k in enumerate(args.pairs):
        p = run["pairs"][k]
        cut = split_node(tree, np.asarray(p["raw_i"]), np.asarray(p["raw_j"]))
        assert int(tree.tree_.feature[cut]) == p["cut_feature"], "arvore reajustada difere da gravada"
        geom = draw_tree(axes[r, 0], tree, p, feats, classes, colors)
        draw_method(axes[r, 1], p, feats, geom)
    T.panel_tags(axes, inside=True)
    fig.tight_layout(w_pad=0.5, h_pad=1.2)
    T.save(fig, args.out)
    print(f"pronto: {args.out}")


if __name__ == "__main__":
    main()

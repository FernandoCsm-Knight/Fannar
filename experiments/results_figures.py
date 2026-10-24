"""Figuras compostas da Secao de Resultados, a partir das saidas ja gravadas.

  res_comportamento  P1 e P4 nas quatro familias (bases tabulares, Pet com ResNet e com ViT,
                     CIFAR-100): uma leitura por painel, uma linha por familia (Gram conjunta, com a
                     faixa de um desvio) e o piso cruzado de cada familia tracejado na mesma cor.
                     Substitui as tres figuras de sete paineis de cada suite.
  res_pares          P6 nos dois estudos de caso: (a, b) curvas de trocas na `letter` e na
                     `pendigits`, (c) atributo da arvore de decisao entre os tres primeiros de cada
                     ranking, nas duas bases, e (d) area de trocas contra o numero de avaliacoes do
                     modelo na `letter`. Substitui as figuras de cada estudo e a do orcamento.

Uso:
    python results_figures.py        # grava em ../docs/figs/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src import theme as T

T.use()

from src.plots_case import STYLE, _key, _present  # noqa: E402

FAMILIES = (  # (pasta, rotulo, serie do tema, marcador)
    ("tabular_suite", "bases tabulares (MLP)", "joint", "o"),
    ("pets_suite", "Pet, ResNet", "activations", "s"),
    ("pets_suite_vit", "Pet, ViT", "grads", "^"),
    ("cifar100_suite", "CIFAR-100, ResNet", "params", "D"),
)
READINGS = (
    ("knn_pred", "$k$-vizinhos contra a predição"),
    ("correctness_auc", "AUC da margem para o acerto"),
    ("confusion_soft", "Spearman com a confusão suave"),
    ("containment_k", r"contenção em $V_r$ ($r = Q-1$)"),
    ("e_ratio_auc_k", r"AUC de $\widehat{E}_{\mathrm{ratio}}$ para o erro"),
)
CASE_KINDS = ("method", "shap_baseline", "ig", "lime", "shap", "absdiff", "dice", "gradxdelta", "random", "floor")


def _curve(summary: dict, key: str):
    """Media e desvio: entre bases, nas tabulares; entre sementes, numa base so."""
    if len(summary["datasets"]) == 1:
        m, s = summary["per_dataset"][summary["datasets"][0]][key]
    else:
        m, s = summary["across"][key][:2]
    return np.asarray(m, dtype=float), np.asarray(s, dtype=float)


def behaviour(out: Path, root: Path) -> None:
    fams = [(json.loads((root / d / "summary_fixed.json").read_text(encoding="utf-8")), lab, key, mk)
            for d, lab, key, mk in FAMILIES if (root / d / "summary_fixed.json").exists()]
    blocks = fams[0][0]["blocks"]
    x = np.arange(len(blocks))
    fig, axes = T.grid(2, 3, width="full", height=4.6)
    flat = axes.ravel()
    for ax, (reading, title) in zip(flat, READINGS):
        for summary, label, key, marker in fams:
            color = T.COLOR[key]
            m, s = _curve(summary, f"{reading}/joint")
            if np.all(~np.isfinite(m)):
                continue
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.12, linewidth=0)
            ax.plot(x, m, lw=1.4, ms=3.0, color=color, marker=marker, markeredgewidth=0.0, label=label)
            f, _ = _curve(summary, f"{reading}/floor")
            ax.plot(x, f, lw=0.9, color=color, linestyle=(0, (3.0, 2.0)), alpha=0.8)
        ax.set_title(title, fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(blocks, fontsize=6.5)
        T.despine(ax)
        T.decimal(ax, "y", 2)
    # o sexto painel e a legenda: familias (linha cheia) e o piso de cada uma (tracejado)
    leg = flat[-1]
    leg.axis("off")
    handles, labels = flat[0].get_legend_handles_labels()
    import matplotlib.lines as mlines
    handles.append(mlines.Line2D([], [], color=T.INK_SOFT, lw=0.9, linestyle=(0, (3.0, 2.0))))
    labels.append("piso cruzado (mesma cor)")
    leg.legend(handles, labels, loc="center", fontsize=7.0, frameon=False)
    T.panel_tags(axes[:, :].ravel()[:5])
    fig.tight_layout()
    T.save(fig, out / "res_comportamento.pdf")


def _tree_top3(runs: list[dict], block: str) -> dict:
    """Fracao dos pares com o atributo da arvore nos rotulos entre os 3 primeiros, por semente."""
    from letter_case_study import local_tree_metrics
    m = local_tree_metrics(runs, "label_tree", block)["metrics"]
    return {k: np.asarray(m[k]["top3"], dtype=float) for k in m}


def pairs(out: Path, root: Path, block: str) -> None:
    import letter_case_study as L

    runs = {b: [json.loads(f.read_text(encoding="utf-8")) for f in sorted((root / f"{b}_case" / "runs").glob("seed*.json"))]
            for b in ("letter", "pendigits")}
    budget = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((root / "letter_budget" / "runs").glob("*.json"))]
    fig, axes = T.grid(2, 2, width="full", height=5.0)

    for ax, base in ((axes[0, 0], "letter"), (axes[0, 1], "pendigits")):
        rs = runs[base]
        grid = np.asarray(rs[0]["grid"])
        for kind in _present(rs, CASE_KINDS):
            style = {**STYLE[kind], "label": L.LABEL[kind]}
            style.pop("marker", None)
            ax.plot(grid, np.mean([r["curves"][_key(kind, block)] for r in rs], 0), lw=1.3, **style)
        ax.set_title(f"\\texttt{{{base}}}" if False else base, fontsize=7.5)
        ax.set_xlabel("fração dos atributos de j trocados em i")
        ax.set_ylabel("fração dos pares na classe de j")
        T.despine(ax)
        T.decimal(ax, "both", 1)

    # (c) arvore: um marcador por base para cada ranking
    ax = axes[1, 0]
    tops = {b: _tree_top3(runs[b], block) for b in runs}
    kinds = [k for k in CASE_KINDS if all(k in tops[b] for b in tops)]
    y = np.arange(len(kinds))[::-1]
    for off, base, marker in ((0.15, "letter", "o"), (-0.15, "pendigits", "^")):
        for yi, k in zip(y, kinds):
            v = tops[base][k]
            color = STYLE[k]["color"]
            ax.plot([v.mean()], [yi + off], linestyle="none", marker=marker, ms=4.2, color=color,
                    markerfacecolor=color if base == "letter" else "white", markeredgewidth=0.9)
            ax.plot([v.mean() - v.std(), v.mean() + v.std()], [yi + off] * 2, color=color, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([L.LABEL[k] for k in kinds], fontsize=6.0)
    ax.set_xlabel("pares com o atributo da árvore entre os 3 primeiros")
    import matplotlib.lines as mlines
    ax.legend([mlines.Line2D([], [], color=T.INK_SOFT, marker="o", linestyle="none", ms=4),
               mlines.Line2D([], [], color=T.INK_SOFT, marker="^", markerfacecolor="white", linestyle="none", ms=4)],
              ["letter", "pendigits"], fontsize=6.0, loc="upper left", frameon=False)
    T.despine(ax)
    T.decimal(ax, "x", 1)

    # (d) orcamento: o metodo, de custo fixo, contra o Shapley amostrado e o LIME
    ax = axes[1, 1]
    budgets, lb = budget[0]["budgets"], budget[0]["lime_budgets"]
    cost = budget[0]["method_cost"]["forward_equivalent"]
    m = np.array([r["auc"]["method"] for r in budget])
    st = STYLE["method"]
    ax.axhspan(m.mean() - m.std(), m.mean() + m.std(), color=st["color"], alpha=0.12, lw=0)
    ax.errorbar([cost], [m.mean()], yerr=[m.std()], fmt="o", ms=5, capsize=2.0, color=st["color"], zorder=5)
    for prefix, bs, kind, name in (("shap", budgets, "shap_baseline", "Shapley amostrado"),
                                   ("lime", lb, "lime", "LIME contrastivo")):
        v = np.array([[r["auc"][f"{prefix}@{b}"] for b in bs] for r in budget])
        c = STYLE[kind]["color"]
        ax.fill_between(bs, v.mean(0) - v.std(0), v.mean(0) + v.std(0), color=c, alpha=0.14, lw=0)
        ax.plot(bs, v.mean(0), lw=1.2, ms=3.2, color=c, marker=STYLE[kind].get("marker", "o"), markeredgewidth=0.0)
        ax.text(bs[-1], v.mean(0)[-1] - 0.012, name, color=c, fontsize=6.0, ha="right", va="top")
    ax.text(cost * 1.15, m.mean() + m.std() + 0.004, "método", color=st["color"], fontsize=6.0, ha="left", va="bottom")
    ax.axvline(cost, color=T.RULE, lw=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("avaliações do modelo por par (letter)")
    ax.set_ylabel("área sob a curva de trocas")
    T.despine(ax)
    T.decimal(ax, "y", 2)

    T.panel_tags(axes)
    fig.tight_layout()
    # legenda das curvas (vale para a, b e, nas cores, para c e d) na margem esquerda de (a), que os
    # rotulos longos de (c) ja reservam na coluna da esquerda
    a = axes[0, 0]
    handles, names = a.get_legend_handles_labels()
    label = a.yaxis.label.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    box = a.get_position()
    fig.legend(handles, names, fontsize=6.0, loc="center right", frameon=False, handlelength=1.6,
               bbox_to_anchor=(label.x0 - 0.01, box.y0 + box.height / 2))
    T.save(fig, out / "res_pares.pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--block", default="b3")
    ap.add_argument("--out", default="../docs/figs")
    args = ap.parse_args()
    out, root = Path(args.out), Path(args.outputs)
    out.mkdir(parents=True, exist_ok=True)
    behaviour(out, root)
    pairs(out, root, args.block)
    print(f"pronto: {out}/res_comportamento.pdf, res_pares.pdf")


if __name__ == "__main__":
    main()

"""Figuras compostas da Secao de Resultados, a partir das saidas ja gravadas.

  res_comportamento  P1 e P4 nas quatro familias (bases tabulares, Pet com ResNet e com ViT,
                     CIFAR-100): uma leitura por painel, uma linha por familia (Gram conjunta, com a
                     faixa de um desvio) e o piso cruzado de cada familia tracejado na mesma cor.
                     Substitui as tres figuras de sete paineis de cada suite.
  res_pares          P6 nos dois estudos de caso, `letter` (decisao nitida) e HELOC (decisao fraca):
                     (a, b) curvas de trocas no MLP, (c) atributo em que a arvore nos rotulos separa i
                     de j entre os tres primeiros de cada ranking, (d) atributos alterados pela
                     construcao do contrafactual, no MLP e na arvore, contra o minimo exato.

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

from src.plots_case import STYLE  # noqa: E402

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
CASES = (("letter", "letter"), ("heloc", "HELOC"))
CASE_KINDS = ("method", "shap_baseline", "ig", "shap", "lime", "absdiff", "dice", "gradxdelta", "random", "floor")
CASE_LABEL = {"method": "método", "shap_baseline": "Shapley exato (j de referência)", "ig": "gradientes integrados",
              "shap": "SHAP contrastivo", "lime": "LIME contrastivo", "absdiff": "|Δx|", "dice": "DiCE",
              "gradxdelta": "grad × Δx", "random": "aleatória", "floor": "método, rede não treinada"}
# (chave nos JSONs do MLP, chave nos da arvore, rotulo, cor)
CF_PROCS = (("minimal", "minimal", "mínimo exato", T.INK_FAINT),
            ("seq_geo", "seq_geo", "sequencial, geometria", T.COLOR["joint"]),
            ("seq_out", "seq_out", "sequencial, saída do modelo", T.COLOR["params"]),
            ("fixed_geo", "fixed_geo", "ranking fixo, geometria", T.COLOR["joint_root"]),
            ("dice_best", None, "DiCE, o mais esparso", STYLE["dice"]["color"]))


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


def _load_runs(folder: Path) -> list[dict]:
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted((folder / "runs").glob("seed*.json"))]


def _tree_top3(runs: list[dict], key: str, block: str) -> np.ndarray:
    """Fracao, por semente, dos pares em que o atributo da arvore nos rotulos esta entre os 3 primeiros."""
    order_key = f"{key}@{block}" if key in ("method", "floor") else key
    out = []
    for r in runs:
        hits = []
        for p in r["pairs"]:
            t = p["trees"]["label_tree"]
            if not t["reproduces"] or t["feature"] is None:
                continue
            order = np.asarray(p["orders"][order_key])
            hits.append(int(np.flatnonzero(order == t["feature"])[0]) < 3)
        out.append(np.mean(hits))
    return np.asarray(out)


def _chance_top3(runs: list[dict]) -> np.ndarray:
    return np.asarray([np.mean([min(3, p["n_differ"]) / p["n_differ"] for p in r["pairs"]
                                if p["trees"]["label_tree"]["reproduces"] and p["trees"]["label_tree"]["feature"] is not None])
                       for r in runs])


def _cf_changed(runs: list[dict], proc: str, tree: bool) -> np.ndarray:
    """Atributos alterados nos contrafactuais validos, media por semente."""
    out = []
    for r in runs:
        vals = []
        for p in r["pairs"]:
            if tree:
                v = p["all"]["k_min_mu"] if proc == "minimal" else (
                    p["all"]["variants"][proc]["n_changed"] if p["all"]["variants"][proc]["valid"] else None)
            else:
                res = p["cf"]["all"].get(proc)
                v = None if res is None else (res["n_changed"] if proc == "minimal" else
                                              (res["n_changed"] if res.get("valid") else None))
            if v is not None:
                vals.append(v)
        out.append(np.mean(vals) if vals else np.nan)
    return np.asarray(out)


def pairs(out: Path, root: Path, block: str) -> None:
    import matplotlib.lines as mlines
    from matplotlib.ticker import MaxNLocator

    mlp = {b: _load_runs(root / f"{b}_case") for b, _ in CASES}
    tree = {b: _load_runs(root / f"tree_counterfactual_{b}") for b, _ in CASES}
    fig, axes = T.grid(2, 2, width="full", height=5.2)

    # (a, b) curvas de trocas no MLP
    for ax, (base, title) in zip((axes[0, 0], axes[0, 1]), CASES):
        rs = mlp[base]
        grid = np.asarray(rs[0]["grid"])
        for kind in CASE_KINDS:
            key = f"{kind}@{block}" if kind in ("method", "floor") else kind
            if key not in rs[0]["curves"]:
                continue
            style = {k: v for k, v in STYLE[kind].items() if k != "marker"}
            style["label"] = CASE_LABEL[kind]
            lw = 1.8 if kind == "method" else 1.1
            ax.plot(grid, np.mean([r["curves"][key] for r in rs], 0), lw=lw, **style)
        ax.set_title(f"{title}, MLP", fontsize=7.5)
        ax.set_xlabel("fração dos atributos de j trocados em i")
        ax.set_ylabel("fração dos pares na classe de j")
        T.despine(ax)
        T.decimal(ax, "both", 1)

    # (c) arvore nos rotulos: o atributo do corte entre os 3 primeiros
    ax = axes[1, 0]
    kinds = [k for k in CASE_KINDS if k not in ("floor",)]
    y = np.arange(len(kinds) + 1)[::-1]
    for off, (base, _), marker in zip((0.15, -0.15), CASES, ("o", "^")):
        rows = [(_tree_top3(mlp[base], k, block), STYLE[k]["color"]) for k in kinds]
        rows.append((_chance_top3(mlp[base]), T.CONTROL))
        for yi, (v, color) in zip(y, rows):
            ax.plot([v.mean()], [yi + off], linestyle="none", marker=marker, ms=4.2, color=color,
                    markerfacecolor=color if base == "letter" else "white", markeredgewidth=0.9)
            ax.plot([v.mean() - v.std(), v.mean() + v.std()], [yi + off] * 2, color=color, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([CASE_LABEL[k] for k in kinds] + ["acaso"], fontsize=6.0)
    ax.set_xlim(0, 1)
    ax.set_xlabel("pares com o atributo da árvore entre os 3 primeiros")
    ax.legend([mlines.Line2D([], [], color=T.INK_SOFT, marker="o", linestyle="none", ms=4),
               mlines.Line2D([], [], color=T.INK_SOFT, marker="^", markerfacecolor="white", linestyle="none", ms=4)],
              [title for _, title in CASES], fontsize=6.0, loc="lower right", frameon=False)
    T.despine(ax)
    T.decimal(ax, "x", 1)

    # (d) atributos alterados pela construcao do contrafactual
    ax = axes[1, 1]
    settings = [(base, title, False) for base, title in CASES] + [(base, title, True) for base, title in CASES]
    width = 0.8 / len(CF_PROCS)
    for si, (base, title, is_tree) in enumerate(settings):
        runs = tree[base] if is_tree else mlp[base]
        for pi, (key_mlp, key_tree, label, color) in enumerate(CF_PROCS):
            key = key_tree if is_tree else key_mlp
            if key is None:
                continue
            v = _cf_changed(runs, key, is_tree)
            ax.bar(si + (pi - (len(CF_PROCS) - 1) / 2) * width, np.nanmean(v), width * 0.92, yerr=np.nanstd(v),
                   color=color, error_kw={"lw": 0.6, "capsize": 1.2}, label=label if si == 0 else None)
    ax.set_xticks(np.arange(len(settings)))
    ax.set_xticklabels([f"{t}\n{'árvore' if tr else 'MLP'}" for _, t, tr in settings], fontsize=6.3)
    ax.set_ylabel("atributos alterados")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=5.8, loc="upper left", frameon=False)
    T.despine(ax)

    T.panel_tags(axes)
    fig.tight_layout()
    # a legenda das curvas (vale para a e b) na margem esquerda de (a), que os rotulos longos de (c)
    # ja reservam na coluna da esquerda
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

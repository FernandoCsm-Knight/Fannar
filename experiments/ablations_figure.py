"""Figura das ablacoes (`res_ablacoes.pdf`), a partir das saidas ja gravadas.

  (a) AUC da margem para o acerto por bloco, bases tabulares: o produto de Hadamard contra a
      soma (= concatenacao), o produto sem T_a, sem J, sem os dois, a fonte A isolada e o piso
      cruzado do produto;
  (b) o mesmo no Oxford-IIIT Pet (ResNet; o transformador se a ResNet nao tiver sido rodada);
  (c) previsao do acerto: a margem geometrica contra os sinais de confianca do item (b);
  (d) verdade de base sintetica: fracao dos k primeiros atributos de cada ranking que sao de
      fato informativos, com o acaso tracejado.

Um painel cujos dados ainda nao existem fica vazio, com o aviso, e os demais sao desenhados.

Uso:
    python ablations_figure.py                # grava ../docs/figs/res_ablacoes.pdf e .png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src import theme as T

T.use()

from src.ablation import CONFIDENCE, LABEL  # noqa: E402

STYLE = {
    "joint": T.series("joint"),
    "additive": {**T.series("joint_root"), "label": LABEL["additive"]},
    "no_angular": {**T.series("params"), "label": LABEL["no_angular"]},
    "no_shift": {**T.series("grads"), "label": LABEL["no_shift"]},
    "no_both": {**T.series("rsa_euclid"), "label": LABEL["no_both"]},
    "activations": T.series("activations"),
}
STYLE["joint"]["label"] = LABEL["joint"]
SYNTH = ("method", "shap_baseline", "ig", "lime", "dice", "gradxdelta", "absdiff", "floor")
SYNTH_LABEL = {"method": "método", "shap_baseline": "Shapley exato", "ig": "grad. integrados",
               "lime": "LIME contr.", "dice": "DiCE", "gradxdelta": "grad × Δx", "absdiff": "|Δx|",
               "floor": "método (não treinada)"}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _missing(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=T.INK_FAINT, fontsize=7)
    T.bare(ax)


def composition_panel(ax, summary: dict | None, name: str) -> None:
    if summary is None:
        _missing(ax, f"sem dados: outputs/ablations/{name}")
        return
    blocks = summary["blocks"]
    x = np.arange(len(blocks))
    for rep, style in STYLE.items():
        m, s = (np.asarray(v, dtype=float) for v in summary["mean"][f"correctness_auc/{rep}"])
        lw = 1.7 if rep == "joint" else 1.1
        if rep == "joint":
            ax.fill_between(x, m - s, m + s, color=style["color"], alpha=0.15, linewidth=0)
        ax.plot(x, m, lw=lw, ms=3.0, **style)
    m = np.asarray(summary["floor"]["correctness_auc/joint"][0], dtype=float)
    ax.plot(x, m, lw=1.0, ms=3.0, **T.control("piso cruzado do produto"))
    ax.set_xticks(x)
    ax.set_xticklabels(blocks)
    ax.set_ylabel("AUC da margem para o acerto")
    T.despine(ax)
    T.decimal(ax, "y", 2)


def confidence_panel(ax, summaries: dict) -> None:
    present = {k: v for k, v in summaries.items() if v is not None}
    if not present:
        _missing(ax, "sem dados: outputs/ablations/")
        return
    names = list(present)
    kinds = ["geo_mid", "geo_last", *CONFIDENCE]
    labels = {"geo_mid": "margem geom. (bloco do meio)", "geo_last": "margem geom. (último bloco)"} | LABEL
    colors = {"geo_mid": T.COLOR["joint"], "geo_last": T.COLOR["joint_root"],
              "softmax": T.COLOR["activations"], "logit_margin": T.COLOR["grads"],
              "trust_score": T.COLOR["params"], "mahalanobis": T.COLOR["rsa_euclid"]}
    width = 0.8 / len(kinds)
    for gi, name in enumerate(names):
        s = present[name]
        blocks = s["blocks"]
        vals = {"geo_mid": s["geometric"][blocks[len(blocks) // 2]], "geo_last": s["geometric"][blocks[-1]]}
        vals |= {k: s["confidence"][k] for k in CONFIDENCE}
        for ki, k in enumerate(kinds):
            m, sd = vals[k]
            ax.bar(gi + (ki - (len(kinds) - 1) / 2) * width, m, width * 0.92, yerr=sd, color=colors[k],
                   error_kw={"lw": 0.6, "capsize": 1.2}, label=labels[k] if gi == 0 else None)
    ax.axhline(0.5, color=T.CONTROL, lw=0.8, linestyle=(0, (3.5, 2.5)))
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([{"tabular": "tabulares", "pets_resnet": "Pet, ResNet",
                         "pets_vit": "Pet, ViT", "cifar100_resnet": "CIFAR-100"}.get(n, n) for n in names])
    ax.set_ylabel("AUC para o acerto")
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=5.8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12), frameon=False)
    T.despine(ax)
    T.decimal(ax, "y", 1)


def synthetic_panel(ax, runs: list[dict]) -> None:
    if not runs:
        _missing(ax, "sem dados: outputs/synthetic_counterfactual")
        return
    keys = [k for k in SYNTH if k in runs[0]["precision"]]
    vals = np.array([[r["precision"][k] for k in keys] for r in runs])
    y = np.arange(len(keys))[::-1]
    for yi, k, m, s in zip(y, keys, vals.mean(0), vals.std(0)):
        color = T.COLOR["joint"] if k == "method" else (T.CONTROL if k == "floor" else T.INK_SOFT)
        ax.barh(yi, m, xerr=s, color=color, height=0.65, error_kw={"lw": 0.6, "capsize": 1.2})
    chance = runs[0]["chance"]
    ax.axvline(chance, color=T.CONTROL, lw=0.9, linestyle=(0, (3.5, 2.5)))
    ax.text(chance, y[0] + 0.5, " acaso", color=T.INK_FAINT, fontsize=6.5, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([SYNTH_LABEL[k] for k in keys], fontsize=6.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"fração dos {runs[0]['n_informative']} primeiros que são informativos")
    T.despine(ax)
    T.decimal(ax, "x", 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablations", default="outputs/ablations")
    ap.add_argument("--synthetic", default="outputs/synthetic_counterfactual")
    ap.add_argument("--out", default="../docs/figs/res_ablacoes.pdf")
    args = ap.parse_args()

    root = Path(args.ablations)
    summaries = {k: _load(root / k / "summary.json")
                 for k in ("tabular", "pets_resnet", "pets_vit", "cifar100_resnet")}
    synth = [json.loads(f.read_text(encoding="utf-8"))
             for f in sorted(Path(args.synthetic, "runs").glob("*.json"))]

    fig, axes = T.grid(2, 2, width="full", height=6.2)
    composition_panel(axes[0, 0], summaries["tabular"], "tabular")
    image = "pets_resnet" if summaries["pets_resnet"] is not None else "pets_vit"
    composition_panel(axes[0, 1], summaries[image], image)
    confidence_panel(axes[1, 0], summaries)
    synthetic_panel(axes[1, 1], synth)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6.3, frameon=False,
                   bbox_to_anchor=(0.5, 1.0))
    T.panel_tags(axes)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    T.save(fig, args.out)
    print(f"pronto: {args.out}")


if __name__ == "__main__":
    main()

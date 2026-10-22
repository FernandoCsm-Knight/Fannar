"""Figura da P9 (`res_semantica.pdf`): as distancias refletem uma hierarquia semantica externa?

Um painel por base e arquitetura: AUC com que a proximidade media entre duas classes separa os
pares do mesmo grupo semantico (especie no Pet, superclasse no CIFAR-100), por bloco, para a
Gram conjunta, a fonte de ativacoes e o piso cruzado (alvo fixo). A linha horizontal e a mesma
AUC calculada sobre a confusao suave do proprio modelo: e o quanto do comportamento ja carrega
a hierarquia, e uma geometria que so a reproduz nao acrescenta nada a ele.

Uso:
    python semantics_figure.py          # grava ../docs/figs/res_semantica.pdf e .png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src import theme as T

T.use()

PANELS = (("outputs/pets_suite", "Pet (espécie), ResNet"), ("outputs/pets_suite_vit", "Pet (espécie), ViT"),
          ("outputs/cifar100_suite", "CIFAR-100 (superclasse), ResNet"))


def _runs(path: str) -> list[dict]:
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(Path(path, "runs").glob("*.json"))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="fixed")
    ap.add_argument("--out", default="../docs/figs/res_semantica.pdf")
    args = ap.parse_args()

    present = [(p, t, _runs(p)) for p, t in PANELS]
    present = [x for x in present if x[2]]
    if not present:
        raise SystemExit("nenhuma suíte de imagens em outputs/")
    fig, axes = T.grid(1, len(present), width="full", height=2.5, sharey=True)
    for ax, (_, title, runs) in zip(axes[0], present):
        blocks = runs[0]["blocks"]
        x = np.arange(len(blocks))
        v = [r["variants"][args.variant] for r in runs]
        series = (("joint", T.series("joint"), lambda m, b: m["metrics"][b]["joint"]),
                  ("activations", T.series("activations"), lambda m, b: m["metrics"][b]["activations"]),
                  ("floor", T.control("piso cruzado"), lambda m, b: m["floor"][b]))
        for _, style, get in series:
            vals = np.array([[get(m, b).get("semantic_auc", np.nan) for b in blocks] for m in v], dtype=float)
            mean, sd = np.nanmean(vals, 0), np.nanstd(vals, 0)
            ax.fill_between(x, mean - sd, mean + sd, color=style["color"], alpha=0.12, linewidth=0)
            ax.plot(x, mean, lw=1.4, ms=3.0, **style)
        behaviour = np.mean([r["behaviour_semantic_auc"] for r in runs])
        ax.axhline(behaviour, color=T.INK_SOFT, lw=0.9, linestyle=(0, (1.0, 1.6)),
                   label="confusão do próprio modelo")
        ax.axhline(0.5, color=T.RULE, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(blocks)
        ax.set_title(title, fontsize=7.5)
        T.despine(ax)
        T.decimal(ax, "y", 2)
    axes[0, 0].set_ylabel("AUC para o mesmo grupo semântico")
    axes[0, 0].legend(fontsize=6.0, loc="upper left")
    T.panel_tags(axes)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    T.save(fig, args.out)
    print(f"pronto: {args.out}")


if __name__ == "__main__":
    main()

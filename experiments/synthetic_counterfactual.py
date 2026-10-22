"""Leitura de pares com verdade de base conhecida (item (d) da Subsecao `subsec:adicionais`).

O teste de trocas da `letter` mede exatamente o jogo que o Shapley com j de referencia
decompoe, e por isso o favorece por construcao; a unica verificacao independente era a arvore.
Aqui a referencia e externa ao modelo e a qualquer ranking: dados sinteticos em que se sabe
quais atributos determinam o rotulo.

Dados. `make_classification` com F atributos, dos quais os `--informative` primeiros sao os
unicos que entram na geracao do rotulo (sem atributos redundantes nem repetidos, `shuffle=False`);
os demais sao ruido gaussiano independente do rotulo. Um MLP residual com a receita de
`tabular_suite.py` e treinado sobre eles.

Pares e rankings como em `letter_case_study.py`: i acertada pelo modelo, j a amostra do teste
mais proxima nos atributos com outra predicao. Rankings comparados:

  method         Δd_f na Gram conjunta (alvo fixo), em cada bloco
  shap_baseline  Shapley exato com j de referencia (2^F avaliacoes)
  ig             gradientes integrados da margem f_{c_j} − f_{c_i} no caminho reto de x_i a x_j
                 (Sundararajan et al., 2017); somam m(x_j) − m(x_i), como o Shapley
  lime           LIME contrastivo (regressao sobre a margem em torno de x_i)
  dice           DiCE (Mothilal et al., 2020), metodo `random`: frequencia com que cada atributo
                 e alterado nos contrafactuais gerados para levar i a classe de j (opcional,
                 exige `dice-ml`)
  gradxdelta, absdiff, random, floor   as referencias de sempre

Duas medidas:

  verdade de base  fracao dos `--informative` primeiros do ranking que sao atributos
                   informativos (precisao@k, k = numero de informativos); o acaso e k/F;
  trocas           a area sob a curva do teste contrafactual de `tabular_differences.py`,
                   para comparar com a `letter`.

Uso:
    python synthetic_counterfactual.py --seeds 5 --pairs 100
    python synthetic_counterfactual.py --seeds 5 --pairs 100 --dice
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_classification

from letter_case_study import baseline_shapley, lime_contrastive
from src.attributions import build_dice, dice_scores, integrated_gradients
from src.data_tabular import split, standardize
from src.mlp import ResidualMLP
from tabular_differences import area, flip_curve, gradxdelta, method_scores
from tabular_suite import train

RANKINGS = ("method", "shap_baseline", "ig", "lime", "dice", "gradxdelta", "absdiff", "random", "floor")
LABEL = {
    "method": r"método ($\Delta_f$, Gram conjunta)",
    "shap_baseline": "Shapley exato com j de referência",
    "ig": "gradientes integrados (x_i → x_j)",
    "lime": "LIME contrastivo",
    "dice": "DiCE (frequência de alteração)",
    "gradxdelta": "grad × Δx",
    "absdiff": "\\|Δx\\|",
    "random": "aleatória",
    "floor": "método na rede não treinada",
}


def make_data(seed: int, args) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=args.n, n_features=args.features, n_informative=args.informative, n_redundant=0,
        n_repeated=0, n_classes=args.classes, n_clusters_per_class=1, class_sep=args.class_sep,
        shuffle=False, random_state=seed)
    return X.astype(np.float32), y.astype(np.int64)


def run_one(seed: int, args, device) -> dict:
    X, y = make_data(seed, args)
    tr, va, te = split(y, seed)
    X_tr, X_va, X_te = standardize(X[tr], X[va], X[te])
    y_tr, y_va, y_te = y[tr], y[va], y[te]
    torch.manual_seed(seed)
    model = ResidualMLP(X.shape[1], args.classes, args.width, args.blocks).to(device)
    untrained = copy.deepcopy(model)
    train(model, X_tr, y_tr, X_va, y_va, device, args, seed)
    model.eval()
    untrained.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_te).to(device)).argmax(1).cpu().numpy()

    lime_explainer = None
    if not args.no_lime:
        from lime.lime_tabular import LimeTabularExplainer
        lime_explainer = LimeTabularExplainer(X_tr, mode="regression", discretize_continuous=False,
                                              random_state=seed)
    dice = build_dice(model, X_tr, y_tr, device, seed) if args.dice else None

    rng = np.random.default_rng(seed)
    F, k = X.shape[1], args.informative
    informative = np.arange(k)
    grid = np.linspace(0, 1, args.steps + 1)
    blocks = model.block_names
    hits = {r: [] for r in RANKINGS} | {f"method@{b}": [] for b in blocks}
    curves = {r: [] for r in RANKINGS} | {f"method@{b}": [] for b in blocks}
    for i in rng.choice(np.flatnonzero(preds == y_te), size=args.pairs, replace=False):
        other = np.flatnonzero(preds != preds[i])
        j = int(other[np.argmin(np.linalg.norm(X_te[other] - X_te[i], axis=1))])
        xi, xj, ci, cj = X_te[i], X_te[j], int(preds[i]), int(preds[j])
        scores = method_scores(model, xi, xj, device, args.fixed_target, args.shift)
        floor = method_scores(untrained, xi, xj, device, args.fixed_target, args.shift)[args.block]
        ranked = {
            "method": scores[args.block],
            "shap_baseline": baseline_shapley(model, xi, xj, ci, cj, device)[0],
            "ig": integrated_gradients(model, xi, xj, ci, cj, device, args.ig_steps),
            "gradxdelta": gradxdelta(model, xi, xj, ci, cj, device),
            "absdiff": np.abs(xj - xi).astype(np.float64),
            "random": rng.random(F),
            "floor": floor,
        } | {f"method@{b}": scores[b] for b in blocks}
        if lime_explainer is not None:
            ranked["lime"] = lime_contrastive(lime_explainer, model, xi, xj, ci, cj, device, args.lime_samples)
        if dice is not None:
            ranked["dice"] = dice_scores(dice, xi, cj, args.dice_cfs, rng)
        for name, score in ranked.items():
            order = np.lexsort((rng.random(F), -np.asarray(score, dtype=float)))
            hits[name].append(float(np.isin(order[:k], informative).mean()))
            curves[name].append(flip_curve(model, xi, xj, cj, order, grid, device))
    present = [r for r in hits if hits[r]]
    mean_curves = {r: np.mean(curves[r], 0).tolist() for r in present}
    return {"seed": seed, "blocks": blocks, "block": args.block, "n_features": F, "n_informative": k,
            "test_acc": float((preds == y_te).mean()), "chance": k / F,
            "precision": {r: float(np.mean(hits[r])) for r in present},
            "precision_pairs": {r: hits[r] for r in present},
            "grid": grid.tolist(), "curves": mean_curves,
            "auc": {r: area(np.asarray(v), grid) for r, v in mean_curves.items()}}


def write_report(runs: list[dict], out: Path) -> None:
    def pm(v):
        v = np.asarray(v, dtype=float)
        return f"{v.mean():.3f} ± {v.std():.3f}".replace(".", ",")

    r0 = runs[0]
    keys = [r for r in RANKINGS if r in r0["precision"]]
    L = ["# Leitura de pares com verdade de base sintética\n",
         f"{len(runs)} sementes; {r0['n_features']} atributos, dos quais {r0['n_informative']} informativos; "
         f"acurácia do MLP {pm([r['test_acc'] for r in runs])}. Método no bloco {r0['block']}. "
         f"Acaso da precisão: {r0['chance']:.3f}.\n".replace(f"{r0['chance']:.3f}", f"{r0['chance']:.3f}".replace(".", ",")),
         "| ranking | precisão@k dos informativos | área de trocas |", "|---|---|---|"]
    for r in keys:
        L.append(f"| {LABEL[r]} | {pm([x['precision'][r] for x in runs])} | {pm([x['auc'][r] for x in runs])} |")
    L += ["", "## O método bloco a bloco\n", "| bloco | precisão@k | área de trocas |", "|---|---|---|"]
    for b in r0["blocks"]:
        L.append(f"| {b} | {pm([x['precision'][f'method@{b}'] for x in runs])} | "
                 f"{pm([x['auc'][f'method@{b}'] for x in runs])} |")
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--features", type=int, default=16, help="16, como na letter: o Shapley exato cabe")
    ap.add_argument("--informative", type=int, default=4)
    ap.add_argument("--classes", type=int, default=6)
    ap.add_argument("--class-sep", type=float, default=1.5)
    ap.add_argument("--block", default="b3")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--ig-steps", type=int, default=64)
    ap.add_argument("--lime-samples", type=int, default=2000)
    ap.add_argument("--no-lime", action="store_true")
    ap.add_argument("--dice", action="store_true", help="inclui o DiCE (pip install dice-ml)")
    ap.add_argument("--dice-cfs", type=int, default=5)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--fixed-target", type=int, default=0)
    ap.add_argument("--out", default="outputs/synthetic_counterfactual")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = []
    for seed in range(args.seeds):
        path = out / "runs" / f"seed{seed}.json"
        if path.exists():
            runs.append(json.loads(path.read_text()))
            continue
        r = run_one(seed, args, device)
        path.write_text(json.dumps(r, ensure_ascii=False))
        runs.append(r)
        p = r["precision"]
        print(f"semente {seed}: precisão@k método {p['method']:.3f} Shapley {p['shap_baseline']:.3f} "
              f"IG {p['ig']:.3f} |Δx| {p['absdiff']:.3f} acaso {r['chance']:.3f}", flush=True)
    write_report(runs, out)
    print(f"pronto: {out}/")


if __name__ == "__main__":
    main()

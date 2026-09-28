"""Exemplo de explicacao em dados tabulares interpretaveis: Titanic (OpenML 40945).

As 10 bases da suite sao multiclasse e tem atributos anonimos (`input12`, `xegvy`), bons para
medir estabilidade e ruins para ler a explicacao. O Titanic serve ao contrario: 11 atributos
com significado, para mostrar *quais caracteristicas distinguem duas amostras para o modelo*.

Nao entra na suite de estabilidade, por dois motivos medidos aqui: e binario (a margem
contrastiva vira a diferenca entre dois logits) e tem poucos atributos, com um deles (sexo)
dominando a decisao -- e com isso o piso da rede nao treinada alcanca o metodo, ou seja, a
base nao separa metodo de referencia. A leitura valida e a explicacao par a par; a validacao
quantitativa continua sendo a das bases multiclasse (`tabular_differences.py`).

Saidas em `outputs/titanic_differences/exemplo.md`:

  agregado      parcela media de cada atributo na aproximacao entre os pares, com |Δx|,
                grad×Δx e a mesma conta na rede nao treinada ao lado;
  pares         exemplos com os valores em unidades originais (anos, libras, sexo) e quantos
                atributos do ranking bastam para i ser classificada como j.

Uso:
    python titanic_example.py --seeds 3 --pairs 50
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from src.data_tabular import class_names, feature_names, load_tabular, split, standardize
from src.mlp import ResidualMLP
from src.report import table
from tabular_differences import flip_curve, gradxdelta, method_scores
from tabular_suite import train

RANKINGS = ("método", "|Δx|", "grad×Δx", "rede não treinada")


def fit(seed: int, args, device):
    """Treina uma semente e devolve o modelo, o controle nao treinado e o teste."""
    X, y = load_tabular("titanic")
    tr, va, te = split(y, seed)
    X_tr, X_va, X_te = (a.astype(np.float32) for a in standardize(X[tr], X[va], X[te]))
    torch.manual_seed(seed)
    model = ResidualMLP(X.shape[1], int(y.max() + 1), args.width, args.blocks).to(device)
    untrained = copy.deepcopy(model)
    train(model, X_tr, y[tr], X_va, y[va], device, args, seed)
    model.eval()
    untrained.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_te).to(device)).argmax(1).cpu().numpy()
    return model, untrained, X_te, X[te], y[te], preds


def pairs_of(X_te, y_te, preds, count: int, seed: int):
    """i correta, j a amostra mais proxima nos atributos brutos com outra predicao."""
    rng = np.random.default_rng(seed)
    for i in rng.choice(np.flatnonzero(preds == y_te), size=count, replace=False):
        other = np.flatnonzero(preds != preds[i])
        j = int(other[np.argmin(np.linalg.norm(X_te[other] - X_te[i], axis=1))])
        yield int(i), j


def rankings_for(model, untrained, xi, xj, ci, cj, args, device, block: str) -> dict:
    return {
        "método": method_scores(model, xi, xj, device, args.fixed_target, args.shift)[block],
        "|Δx|": np.abs(xj - xi),
        "grad×Δx": gradxdelta(model, xi, xj, ci, cj, device),
        "rede não treinada": method_scores(untrained, xi, xj, device, args.fixed_target, args.shift)[block],
    }


def in_units(value: float, name: str) -> str:
    """Valor no formato original, nao padronizado."""
    if name.startswith("sexo"):
        return "mulher" if value > 0.5 else "homem"
    if name.startswith(("embarque", "cabine")):
        return "sim" if value > 0.5 else "não"
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--block", default="b3")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--fixed-target", type=int, default=0)
    ap.add_argument("--out", default="outputs/titanic_differences")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats, classes = feature_names("titanic"), class_names("titanic")
    grid = np.linspace(0, 1, args.steps + 1)
    n_features = len(feats)

    shares = {k: np.zeros(n_features) for k in RANKINGS}
    top1 = {k: np.zeros(n_features) for k in RANKINGS}
    examples, accuracies, n = [], [], 0
    for seed in range(args.seeds):
        model, untrained, X_te, raw_te, y_te, preds = fit(seed, args, device)
        accuracies.append(float((preds == y_te).mean()))
        for i, j in pairs_of(X_te, y_te, preds, args.pairs, seed):
            xi, xj, ci, cj = X_te[i], X_te[j], int(preds[i]), int(preds[j])
            scores = rankings_for(model, untrained, xi, xj, ci, cj, args, device, args.block)
            for key, v in scores.items():
                positive = np.maximum(v, 0)
                shares[key] += positive / max(positive.sum(), 1e-12)
                top1[key][int(np.argmax(v))] += 1
            n += 1
            if seed == 0 and len(examples) < args.examples:
                order = np.argsort(-scores["método"])
                curve = flip_curve(model, xi, xj, cj, order, grid, device)
                hit = np.flatnonzero(curve > 0.5)
                positive = np.maximum(scores["método"], 0)
                examples.append({
                    "i": i, "j": j, "ci": ci, "cj": cj, "true_i": int(y_te[i]), "true_j": int(y_te[j]),
                    "k": int(np.ceil(grid[hit[0]] * n_features)) if len(hit) else None,
                    "rows": [(feats[f], in_units(raw_te[i][f], feats[f]), in_units(raw_te[j][f], feats[f]),
                              positive[f] / max(positive.sum(), 1e-12)) for f in order[: args.top]]})
        print(f"semente {seed}: acerto no teste {accuracies[-1]:.3f}", flush=True)

    order = np.argsort(-shares["método"])
    md = ["# Titanic: quais características distinguem duas amostras para o modelo\n",
          f"MLP residual de {args.blocks} blocos, acerto no teste "
          f"{np.mean(accuracies):.3f} ± {np.std(accuracies):.3f} "
          f"(classe majoritária {max(np.bincount(y_te)) / len(y_te):.3f}); "
          f"{n} pares, Gram conjunta do bloco {args.block}, alvo fixo.\n",
          table("Parcela média da aproximação entre i e j, por atributo",
                {k: {feats[f]: shares[k][f] / n for f in order} for k in RANKINGS},
                [feats[f] for f in order]),
          table("Em que fração dos pares cada atributo é o primeiro do ranking",
                {k: {feats[f]: top1[k][f] / n for f in order} for k in RANKINGS},
                [feats[f] for f in order]),
          "\n## Pares\n"]
    for e in examples:
        k = "nenhum número de trocas bastou" if e["k"] is None else f"bastam os {e['k']} primeiros atributos"
        md += [f"**i**: predita {classes[e['ci']]} (verdade {classes[e['true_i']]}) · "
               f"**j**: predita {classes[e['cj']]} (verdade {classes[e['true_j']]}) · {k}\n",
               "| atributo | i | j | parcela |", "|---|---|---|---|"]
        md += [f"| {name} | {vi} | {vj} | {share:.0%} |" for name, vi, vj, share in e["rows"]]
        md.append("")
    (out / "exemplo.md").write_text("\n".join(md) + "\n")
    print(f"pronto: {out}/exemplo.md")


if __name__ == "__main__":
    main()

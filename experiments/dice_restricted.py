"""DiCE restrito aos atributos que diferem entre i e j, sobre as execucoes ja gravadas do estudo de caso.

O DiCE gera contrafactuais de x_i alterando atributos livremente, inclusive os que ja sao iguais em
x_i e x_j, e a sua pontuacao (fracao dos contrafactuais que alteram o atributo) pode por esses
atributos no topo do ranking -- na `letter`, o 1o atributo do DiCE nao difere entre i e j em 23% dos
pares. No teste de trocas, levar ao valor de j um atributo que ja tem o valor de j nao muda nada, e
o DiCE seria penalizado por algo que as demais tecnicas contrastivas evitam por construcao (elas dao
zero aos atributos que nao diferem) e que a ordem aleatoria tambem evita (ela sorteia so entre os que
diferem). A comparacao justa ordena pelo DiCE apenas os atributos que diferem e poe os demais no fim.

As pontuacoes do DiCE ja estao nas execucoes gravadas; o que falta e o modelo, para refazer as curvas
de troca. Ele e reconstruido como em `letter_budget.py` (mesma semente, mesma ordem de sorteios), e a
area do metodo e conferida contra a gravada antes de qualquer escrita. O resultado substitui `dice`
nas execucoes e a versao sem restricao fica em `dice_all`.

Uso:
    python dice_restricted.py
    FANNAR_CASE=pendigits python dice_restricted.py
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from letter_case_study import DATASET
from src.data_tabular import load_tabular, split, standardize
from src.mlp import ResidualMLP
from tabular_differences import area, flip_curve
from tabular_suite import train


def rebuild(seed: int, args, device):
    """Modelo e dados do estudo de caso, na mesma ordem de sorteios de `letter_case_study.run_one`."""
    X, y = load_tabular(DATASET)
    tr, va, te = split(y, seed)
    X_tr, X_va, X_te = (a.astype(np.float32) for a in standardize(X[tr], X[va], X[te]))
    torch.manual_seed(seed)
    model = ResidualMLP(X.shape[1], int(y.max() + 1), args.width, args.blocks).to(device)
    copy.deepcopy(model)  # o estudo de caso guarda a rede nao treinada aqui; nao altera sorteios
    train(model, X_tr, y[tr], X_va, y[va], device, args, seed)
    model.eval()
    return model, X_te


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=f"outputs/{DATASET}_case/runs")
    ap.add_argument("--block", default="b3")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for path in sorted(Path(args.runs).glob("seed*.json")):
        run = json.loads(path.read_text())
        if "dice_all" in run["auc"]:
            print(f"{path.name}: já restrito")
            continue
        model, X_te = rebuild(run["seed"], args, device)
        grid = np.asarray(run["grid"])
        # conferencia: com o modelo reconstruido, a ordem gravada do metodo reproduz a area gravada
        key = f"method@{args.block}"
        check = [flip_curve(model, X_te[p["i"]], X_te[p["j"]], p["pred_j"], np.asarray(p["orders"][key]), grid, device)
                 for p in run["pairs"]]
        got, want = area(np.mean(check, 0), grid), run["auc"][key]
        assert abs(got - want) < 1e-6, f"{path.name}: modelo reconstruído difere ({got} × {want})"

        curves = []
        for p in run["pairs"]:
            xi, xj = X_te[p["i"]], X_te[p["j"]]
            differ = np.abs(np.asarray(p["raw_j"]) - np.asarray(p["raw_i"])) > 1e-9
            score = np.where(differ, np.asarray(p["dice"]), -1.0)  # os que nao diferem vao para o fim
            order = np.argsort(-score, kind="stable")
            p["orders"]["dice_all"] = p["orders"]["dice"]
            p["orders"]["dice"] = order.tolist()
            curves.append(flip_curve(model, xi, xj, p["pred_j"], order, grid, device))
        run["curves"]["dice_all"], run["auc"]["dice_all"] = run["curves"]["dice"], run["auc"]["dice"]
        run["curves"]["dice"] = np.mean(curves, 0).tolist()
        run["auc"]["dice"] = area(np.asarray(run["curves"]["dice"]), grid)
        path.write_text(json.dumps(run, ensure_ascii=False))
        print(f"{path.name}: DiCE {run['auc']['dice_all']:.3f} -> {run['auc']['dice']:.3f} "
              f"(aleatória entre os que diferem {run['auc']['random']:.3f})", flush=True)


if __name__ == "__main__":
    main()

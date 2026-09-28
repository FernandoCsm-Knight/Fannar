"""Por que esta amostra teve uma classificacao diferente daquela? (dados tabulares)

Em dados tabulares nao ha posicao espacial; a unidade da resposta e o **atributo**. Para um
par (i, j) que o modelo classifica de formas diferentes, a pergunta vira: quais atributos,
se i os tivesse com os valores de j, aproximariam i de j no espaco de representacao?

Atribuicao pelo metodo. Para cada atributo f, x_i^(f) e x_i com o atributo f trocado pelo
valor de x_j, e

    Δd_f = d(i, j) − d(i^(f), j)

na Gram conjunta de um bloco (positivo = a troca aproxima i de j). A distancia par a par sai
das tres similaridades de componente -- k_l = (cos + 1)/2 da eq. `componente` -- como
d² = 2(1 − k_W k_A k_Γ). E exatamente a distancia da Gram transformada a menos de uma escala
global: o centramento e uma translacao comum (nao muda distancias) e o traco so reescala.
Γ usa o **alvo fixo**: com o alvo predito o gradiente carrega a propria predicao, e a
distancia "saberia" quais trocas mudam a classificacao -- o teste abaixo ficaria circular.

Pares. Para cada amostra i do teste, j e a amostra mais proxima de i **nos atributos brutos**
que o modelo classifica de outro jeito: o caso "parecidas, mas classificadas diferente". O par
e escolhido sem usar o metodo, para nao favorece-lo.

Teste contrafactual. Trocam-se os k primeiros atributos de cada ranking (valores de j em i) e
mede-se se a predicao de i passa a ser a de j, ao longo de k; com todos os atributos trocados
i vira j, entao toda curva termina em 1. A area sob a curva (sobre a fracao de atributos) e
maior quando o ranking poe primeiro o que decide. Referencias:

  |Δx|        a maior diferenca bruta entre as duas amostras (padronizadas)
  grad×Δx     ∂(logit_{c_j} − logit_{c_i})/∂x_i · (x_j − x_i), a atribuicao de primeira ordem
  aleatoria   piso por construcao
  piso        o mesmo Δd calculado na rede nao treinada (mesmos pesos iniciais)

Uso:
    python tabular_differences.py --seeds 3
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from src import plots_differences as P
from src.data_tabular import ALL, DATASETS, class_names, feature_names, load_tabular, split, standardize
from src.mlp import ResidualMLP
from tabular_suite import extract, train

RANKINGS = ("method", "absdiff", "gradxdelta", "random", "floor")
LABEL = {"method": "método (Δd na Gram conjunta)", "absdiff": "|Δx|", "gradxdelta": "grad × Δx",
         "random": "ordem aleatória", "floor": "método na rede não treinada"}


def joint_distances(model, rows: np.ndarray, device, target, shift: float) -> dict:
    """d(linha 0, cada linha) na Gram conjunta de cada bloco, a menos de uma escala global."""
    sources = extract(model, rows, device, target)["sources"]
    out = {}
    for b, src in sources.items():
        joint = np.ones(len(rows))
        for key in ("params", "activations", "grads"):
            u = src[key].reshape(len(rows), -1).astype(np.float64)
            norms = np.linalg.norm(u, axis=1)
            cos = (u @ u[0]) / np.maximum(norms * norms[0], 1e-300)
            joint *= (cos + shift) / (1.0 + shift)
        out[b] = np.sqrt(np.maximum(2.0 * (1.0 - joint), 0.0))
    return out


def method_scores(model, xi, xj, device, target, shift) -> dict:
    """Δd_f por bloco: quanto trocar o atributo f (valor de j em i) aproxima i de j."""
    F = len(xi)
    swapped = np.repeat(xi[None], F, axis=0)
    swapped[np.arange(F), np.arange(F)] = xj
    rows = np.concatenate([xj[None], xi[None], swapped]).astype(np.float32)
    d = joint_distances(model, rows, device, target, shift)
    return {b: v[1] - v[2:] for b, v in d.items()}


def gradxdelta(model, xi, xj, ci, cj, device) -> np.ndarray:
    x = torch.from_numpy(xi[None].astype(np.float32)).to(device).requires_grad_(True)
    model.eval()
    logits = model(x)
    (logits[0, cj] - logits[0, ci]).backward()
    return (x.grad[0].cpu().numpy() * (xj - xi)).astype(np.float64)


@torch.no_grad()
def flip_curve(model, xi, xj, cj, order: np.ndarray, grid: np.ndarray, device) -> np.ndarray:
    """Fracao dos passos em que i, com os k primeiros atributos de j, e classificada como j."""
    F = len(xi)
    counts = np.unique(np.round(grid * F).astype(int))
    rank = np.empty(F, dtype=int)
    rank[order] = np.arange(F)
    batch = np.repeat(xi[None], len(counts), axis=0)
    for r, k in enumerate(counts):
        take = rank < k
        batch[r, take] = xj[take]
    preds = model(torch.from_numpy(batch.astype(np.float32)).to(device)).argmax(1).cpu().numpy()
    return np.interp(grid, counts / F, (preds == cj).astype(float))


def area(y: np.ndarray, x: np.ndarray) -> float:
    return float(((y[1:] + y[:-1]) * 0.5 * np.diff(x)).sum())


def run_one(name: str, seed: int, args, device) -> dict:
    X, y = load_tabular(name)
    train_idx, val_idx, test_idx = split(y, seed)
    X_tr, X_va, X_te = standardize(X[train_idx], X[val_idx], X[test_idx])
    y_tr, y_va, y_te = y[train_idx], y[val_idx], y[test_idx]
    n_classes = int(y.max() + 1)
    torch.manual_seed(seed)
    model = ResidualMLP(X.shape[1], n_classes, args.width, args.blocks).to(device)
    untrained = copy.deepcopy(model)
    train(model, X_tr, y_tr, X_va, y_va, device, args, seed)
    model.eval()
    untrained.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_te).to(device)).argmax(1).cpu().numpy()

    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, args.steps + 1)
    blocks = model.block_names
    pairs, curves = [], {r: [] for r in RANKINGS if r not in ("method", "floor")}
    curves_block = {f"method@{b}": [] for b in blocks} | {f"floor@{b}": [] for b in blocks}
    for i in rng.choice(np.flatnonzero(preds == y_te), size=args.pairs, replace=False):
        other = np.flatnonzero(preds != preds[i])
        j = int(other[np.argmin(np.linalg.norm(X_te[other] - X_te[i], axis=1))])
        xi, xj, ci, cj = X_te[i], X_te[j], int(preds[i]), int(preds[j])
        scores = method_scores(model, xi, xj, device, args.fixed_target, args.shift)
        scores_u = method_scores(untrained, xi, xj, device, args.fixed_target, args.shift)
        orders = {
            "absdiff": np.argsort(-np.abs(xj - xi)),
            "gradxdelta": np.argsort(-gradxdelta(model, xi, xj, ci, cj, device)),
            "random": rng.permutation(len(xi)),
        }
        for r, order in orders.items():
            curves[r].append(flip_curve(model, xi, xj, cj, order, grid, device))
        for b in blocks:
            curves_block[f"method@{b}"].append(flip_curve(model, xi, xj, cj, np.argsort(-scores[b]), grid, device))
            curves_block[f"floor@{b}"].append(flip_curve(model, xi, xj, cj, np.argsort(-scores_u[b]), grid, device))
        eb = scores[args.example_block]
        top = np.argsort(-eb)[: args.top]
        pairs.append({"i": int(i), "j": int(j), "true_i": int(y_te[i]), "true_j": int(y_te[j]),
                      "pred_i": ci, "pred_j": cj,
                      "top": [{"feature": int(f), "delta_d": float(eb[f]), "share": float(eb[f] / max(eb[eb > 0].sum(), 1e-12)),
                               "value_i": float(xi[f]), "value_j": float(xj[f])} for f in top],
                      "input_distance": float(np.linalg.norm(xj - xi))})
    mean_curves = {k: np.mean(v, 0).tolist() for k, v in (curves | curves_block).items()}
    return {"dataset": name, "seed": seed, "n_features": int(X.shape[1]), "grid": grid.tolist(),
            "blocks": blocks, "curves": mean_curves,
            "auc": {k: area(np.asarray(v), grid) for k, v in mean_curves.items()}, "pairs": pairs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--examples", type=int, default=3, help="pares explicados por base")
    ap.add_argument("--example-block", default="b3")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--fixed-target", type=int, default=0)
    ap.add_argument("--out", default="outputs/tabular_differences")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = []
    for name in args.datasets:
        for seed in range(args.seeds):
            path = out / "runs" / f"{name}_s{seed}.json"
            if path.exists():
                runs.append(json.loads(path.read_text()))
                continue
            r = run_one(name, seed, args, device)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(r, ensure_ascii=False))
            tmp.replace(path)
            runs.append(r)
            a = r["auc"]
            print(f"{name:14s} semente {seed}: AUC método@{args.example_block} {a[f'method@{args.example_block}']:.3f} "
                  f"|Δx| {a['absdiff']:.3f} grad×Δx {a['gradxdelta']:.3f} aleatória {a['random']:.3f} "
                  f"piso {a[f'floor@{args.example_block}']:.3f}", flush=True)

    write_report(runs, args, out)
    print(f"pronto: {out}/")


def write_report(runs: list[dict], args, out: Path) -> None:
    names = [n for n in ALL if any(r["dataset"] == n for r in runs)]
    blocks = runs[0]["blocks"]
    eb = args.example_block

    def pm(values):
        v = np.asarray(values, dtype=float)
        return f"{v.mean():.3f} ± {v.std():.3f}".replace(".", ",")

    keys = [f"method@{eb}", "absdiff", "gradxdelta", "random", f"floor@{eb}"]
    heads = [f"método ({eb})", "|Δx|", "grad×Δx", "aleatória", f"piso ({eb})"]
    lines = ["# Por que esta amostra teve outra classificação? — dados tabulares\n",
             f"Área sob a curva da fração de pares que passam à classe de j conforme os k primeiros "
             f"atributos de cada ranking são trocados (maior = o ranking põe primeiro o que decide). "
             f"{args.pairs} pares por execução; média ± desvio entre sementes.\n",
             "| base | " + " | ".join(heads) + " |", "|---|" + "---|" * len(heads)]
    per_dataset = {}
    for n in names:
        rs = [r for r in runs if r["dataset"] == n]
        per_dataset[n] = {k: np.mean([r["auc"][k] for r in rs]) for k in runs[0]["auc"]}
        lines.append(f"| {n} | " + " | ".join(pm([r["auc"][k] for r in rs]) for k in keys) + " |")
    lines.append("| **média entre bases** | " + " | ".join(pm([per_dataset[n][k] for n in names]) for k in keys) + " |")
    lines += ["", "## O método bloco a bloco (média ± desvio entre bases)\n",
              "| | " + " | ".join(blocks) + " |", "|---|" + "---|" * len(blocks)]
    for kind, label in (("method", "método"), ("floor", "piso")):
        lines.append(f"| {label} | " + " | ".join(pm([per_dataset[n][f'{kind}@{b}'] for n in names]) for b in blocks) + " |")
    for ref in ("absdiff", "gradxdelta", "random"):
        lines.append(f"| {LABEL[ref]} | " + " | ".join(pm([per_dataset[n][ref] for n in names]) for _ in blocks) + " |")
    lines += ["", f"## Em que fração das {len(runs)} execuções o método supera cada referência\n",
          "Comparação entre as médias de cada execução (50 pares cada), não par a par.\n"]
    for ref in ("absdiff", "gradxdelta", "random", f"floor@{eb}"):
        wins = np.mean([r["auc"][f"method@{eb}"] > r["auc"][ref] for r in runs])
        lines.append(f"- {LABEL.get(ref.split('@')[0], ref)}: {wins:.0%} das execuções")
    (out / "tables.md").write_text("\n".join(lines) + "\n")

    expl = ["# Exemplos de explicação\n",
            f"Para cada par, os {args.top} atributos cuja troca mais aproxima i de j na Gram conjunta do "
            f"bloco {eb}, com os valores padronizados em i e em j e a fração do ganho total de proximidade.\n"]
    for n in names:
        r = next(r for r in runs if r["dataset"] == n and r["seed"] == 0)
        feats, classes = feature_names(n), class_names(n)
        expl.append(f"## {n}\n")
        for p in r["pairs"][: args.examples]:
            expl.append(f"**i**: classe {classes[p['true_i']]}, predita {classes[p['pred_i']]} · "
                        f"**j**: classe {classes[p['true_j']]}, predita {classes[p['pred_j']]}\n")
            expl.append("| atributo | valor em i | valor em j | fração do ganho |\n|---|---|---|---|")
            for t in p["top"]:
                expl.append(f"| {feats[t['feature']]} | {t['value_i']:+.2f} | {t['value_j']:+.2f} | {t['share']:.0%} |")
            expl.append("")
    (out / "explanations.md").write_text("\n".join(expl) + "\n")
    P.plot_tabular_differences(runs, names, blocks, eb, LABEL, str(out / "differences.png"))


if __name__ == "__main__":
    main()

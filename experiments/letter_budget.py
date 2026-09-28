"""Comparacao com o mesmo orcamento de avaliacoes do modelo: metodo × SHAP × LIME na `letter`.

O metodo tem custo fixo por par: F + 2 = 18 linhas numa passagem direta e numa reversa, o que
equivale a ~3(F + 2) = 54 passagens diretas (uma reversa custa cerca de duas diretas). SHAP e
LIME, ao contrario, melhoram com mais avaliacoes. Para comparar com o mesmo custo, os dois sao
rodados com orcamentos B por par e o metodo entra como um ponto fixo no seu custo:

  Shapley (j de referencia)  estimado por permutacoes amostradas (Strumbelj & Kononenko, 2014):
                             cada permutacao custa F + 1 avaliacoes, entao B da floor(B/(F+1))
                             permutacoes; com B = 2^F o valor e o exato de `letter_case_study.py`.
  LIME contrastivo           o mesmo de `letter_case_study.py`, com num_samples = B.

Os pares, os modelos e a arvore sao os de `letter_case_study.py` (mesma semente, mesma ordem de
sorteio); o script confere que a AUC do metodo reproduz a gravada em `outputs/letter_case/runs`.

Saidas em `outputs/letter_budget/`: `runs/seed{s}.json`, `tables.md`, `budget.pdf/png`.

Uso:
    python letter_budget.py --seeds 5 --pairs 100
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from lime.lime_tabular import LimeTabularExplainer
from sklearn.tree import DecisionTreeClassifier

from letter_case_study import DATASET, fmt, lime_contrastive, pm, rank_of, split_node
from src.data_tabular import load_tabular, split, standardize
from src.mlp import ResidualMLP
from tabular_differences import area, flip_curve, method_scores
from tabular_suite import train

BUDGETS = (18, 54, 100, 250, 500, 1000, 5000)


def sampled_shapley(model, x_from, x_to, c_from: int, c_to: int, device, budget: int, rng) -> np.ndarray:
    """Shapley de v(S) = m(x_to em S, x_from fora), por permutacoes: floor(budget/(F+1)) delas."""
    F = len(x_from)
    n_perm = max(1, budget // (F + 1))
    rows, perms = [], []
    for _ in range(n_perm):
        perm = rng.permutation(F)
        cur = x_from.copy()
        rows.append(cur.copy())
        for f in perm:
            cur[f] = x_to[f]
            rows.append(cur.copy())
        perms.append(perm)
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(rows, dtype=np.float32)).to(device))
        v = (logits[:, c_to] - logits[:, c_from]).double().cpu().numpy().reshape(n_perm, F + 1)
    phi = np.zeros(F)
    for k, perm in enumerate(perms):
        phi[perm] += np.diff(v[k])
    return phi / n_perm


def run_one(seed: int, args, device) -> dict:
    # --- mesma construcao de `letter_case_study.run_one` (mesma ordem de sorteios) ---
    X, y = load_tabular(DATASET)
    tr, va, te = split(y, seed)
    X_tr, X_va, X_te = (a.astype(np.float32) for a in standardize(X[tr], X[va], X[te]))
    raw_tr, raw_te = X[tr], X[te]
    torch.manual_seed(seed)
    model = ResidualMLP(X.shape[1], int(y.max() + 1), args.width, args.blocks).to(device)
    copy.deepcopy(model)  # o estudo de caso guarda a rede nao treinada aqui; nao altera sorteios
    train(model, X_tr, y[tr], X_va, y[va], device, args, seed)
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_te).to(device)).argmax(1).cpu().numpy()
    y_te, F = y[te], X.shape[1]
    rng = np.random.default_rng(seed)
    tree = DecisionTreeClassifier(min_samples_leaf=3, random_state=seed).fit(raw_tr, y[tr])
    tree_pred = tree.predict(raw_te)
    pairs = []
    for i in rng.choice(np.flatnonzero(preds == y_te), size=args.pairs, replace=False):
        other = np.flatnonzero(preds != preds[i])
        pairs.append((int(i), int(other[np.argmin(np.linalg.norm(X_te[other] - X_te[i], axis=1))])))

    grid = np.linspace(0, 1, F + 1)
    budgets = [b for b in BUDGETS] + [2 ** F]
    lime_reg = LimeTabularExplainer(X_tr, mode="regression", discretize_continuous=False,
                                    sample_around_instance=True, random_state=seed)
    srng = np.random.default_rng(1000 + seed)   # sorteios proprios, nao mexem nos do estudo de caso
    curves = {"method": []} | {f"shap@{b}": [] for b in budgets} | {f"lime@{b}": [] for b in BUDGETS}
    cut_pos = {k: [] for k in curves}
    times = {k: [] for k in curves}
    for i, j in pairs:
        xi, xj, ci, cj = X_te[i].astype(np.float64), X_te[j].astype(np.float64), int(preds[i]), int(preds[j])
        t0 = time.perf_counter()
        scores = {"method": method_scores(model, xi.astype(np.float32), xj.astype(np.float32), device,
                                          args.fixed_target, args.shift)[args.block]}
        times["method"].append(time.perf_counter() - t0)
        for b in budgets:
            t0 = time.perf_counter()
            scores[f"shap@{b}"] = sampled_shapley(model, xi, xj, ci, cj, device, b, srng)
            times[f"shap@{b}"].append(time.perf_counter() - t0)
        for b in BUDGETS:
            t0 = time.perf_counter()
            scores[f"lime@{b}"] = lime_contrastive(lime_reg, model, xi, xj, ci, cj, device, b)
            times[f"lime@{b}"].append(time.perf_counter() - t0)
        node = split_node(tree, raw_te[i], raw_te[j])
        use_tree = node is not None and tree_pred[i] == ci and tree_pred[j] == cj
        for k, sc in scores.items():
            order = np.argsort(-sc, kind="stable")
            curves[k].append(flip_curve(model, xi.astype(np.float32), xj.astype(np.float32), cj, order, grid, device))
            if use_tree:
                cut_pos[k].append(rank_of(order, int(tree.tree_.feature[node])))
    auc = {k: area(np.mean(v, 0), grid) for k, v in curves.items()}
    return {"seed": seed, "budgets": budgets, "lime_budgets": list(BUDGETS), "auc": auc,
            "tree_top1": {k: float(np.mean(np.asarray(v) == 0)) for k, v in cut_pos.items()},
            "tree_top3": {k: float(np.mean(np.asarray(v) < 3)) for k, v in cut_pos.items()},
            "time_ms": {k: 1e3 * float(np.mean(v)) for k, v in times.items()},
            "method_cost": {"rows": F + 2, "forward_equivalent": 3 * (F + 2)}}


def write_report(runs, out: Path) -> None:
    budgets, lb = runs[0]["budgets"], runs[0]["lime_budgets"]
    L = ["# Mesmo orçamento de avaliações: método × Shapley amostrado × LIME (`letter`)\n",
         f"{len(runs)} sementes; o método custa {runs[0]['method_cost']['rows']} linhas numa passagem direta e numa "
         f"reversa (~{runs[0]['method_cost']['forward_equivalent']} passagens diretas). Média ± desvio entre sementes.\n",
         f"Método: AUC {pm([r['auc']['method'] for r in runs])}, corte da árvore em 1º "
         f"{pm([r['tree_top1']['method'] for r in runs])}, entre os 3 primeiros {pm([r['tree_top3']['method'] for r in runs])}, "
         f"{fmt(np.mean([r['time_ms']['method'] for r in runs]), 1)} ms por par.\n",
         "| orçamento (avaliações por par) | Shapley amostrado: AUC | vence o método | LIME: AUC | vence o método | "
         "Shapley: corte em 1º | LIME: corte em 1º |", "|---|---|---|---|---|---|---|"]
    for b in budgets:
        s = f"shap@{b}"
        lime = f"lime@{b}" if b in lb else None
        row = [f"{b}" + (" (exato)" if b == 2 ** 16 else ""), pm([r["auc"][s] for r in runs]),
               f"{np.mean([r['auc'][s] > r['auc']['method'] for r in runs]):.0%}",
               pm([r["auc"][lime] for r in runs]) if lime else "—",
               f"{np.mean([r['auc'][lime] > r['auc']['method'] for r in runs]):.0%}" if lime else "—",
               pm([r["tree_top1"][s] for r in runs]), pm([r["tree_top1"][lime] for r in runs]) if lime else "—"]
        L.append("| " + " | ".join(row) + " |")
    (out / "tables.md").write_text("\n".join(L) + "\n")


def plot(runs, path: str) -> None:
    """Media ± desvio entre sementes; o metodo, de custo fixo, e o ponto e a faixa horizontal."""
    from src import theme as T
    T.use()
    import matplotlib.pyplot as plt  # noqa: F401
    budgets, lb = runs[0]["budgets"], runs[0]["lime_budgets"]
    cost = runs[0]["method_cost"]
    fig, axes = T.grid(1, 2, width="full", height=2.7)
    for ax, key, ylabel in ((axes[0, 0], "auc", "área sob a curva de trocas"),
                            (axes[0, 1], "tree_top1", "corte da árvore em 1º do ranking")):
        m = np.array([r[key]["method"] for r in runs])
        st = T.series("joint")
        ax.axhspan(m.mean() - m.std(), m.mean() + m.std(), color=st["color"], alpha=0.12, lw=0)
        ax.axhline(m.mean(), color=st["color"], lw=1.0, linestyle=(0, (4, 2)))
        ax.errorbar([cost["forward_equivalent"]], [m.mean()], yerr=[m.std()], fmt="o", ms=5.5, capsize=2.5,
                    color=st["color"], label=f"método (custo fixo, ~{cost['forward_equivalent']} avaliações)", zorder=5)
        for name, bs, series, prefix in (("Shapley amostrado (j de referência)", budgets, "params", "shap"),
                                         ("LIME contrastivo", lb, "joint_root", "lime")):
            v = np.array([[r[key][f"{prefix}@{b}"] for b in bs] for r in runs])
            s2 = T.series(series)
            ax.fill_between(bs, v.mean(0) - v.std(0), v.mean(0) + v.std(0), color=s2["color"], alpha=0.14, lw=0)
            ax.errorbar(bs, v.mean(0), yerr=v.std(0), capsize=2.0, lw=1.2, elinewidth=0.8, ms=3.4,
                        color=s2["color"], marker=s2["marker"], label=name)
        ax.axvline(cost["forward_equivalent"], color=T.RULE, lw=0.8, zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel("avaliações do modelo por par (escala log)")
        ax.set_ylabel(ylabel)
        T.despine(ax)
        T.decimal(ax, "y", 2)
    axes[0, 0].legend(fontsize=6.0, loc="lower right")
    T.panel_tags(axes)
    fig.tight_layout()
    T.save(fig, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--pairs", type=int, default=100)
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
    ap.add_argument("--reference", default="outputs/letter_case/runs")
    ap.add_argument("--out", default="outputs/letter_budget")
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
        t0 = time.perf_counter()
        r = run_one(seed, args, device)
        ref = Path(args.reference) / f"seed{seed}.json"
        if ref.exists():
            want = json.loads(ref.read_text())["auc"][f"method@{args.block}"]
            assert abs(r["auc"]["method"] - want) < 1e-6, f"pares ou modelo diferem do estudo de caso ({r['auc']['method']} × {want})"
        path.write_text(json.dumps(r))
        runs.append(r)
        print(f"semente {seed} ({time.perf_counter() - t0:.0f} s): método {r['auc']['method']:.3f} | Shapley@54 "
              f"{r['auc']['shap@54']:.3f} @5000 {r['auc']['shap@5000']:.3f} | LIME@54 {r['auc']['lime@54']:.3f} "
              f"@5000 {r['auc']['lime@5000']:.3f}", flush=True)
    write_report(runs, out)
    plot(runs, str(out / "budget.png"))
    print(f"pronto: {out}/")


if __name__ == "__main__":
    main()

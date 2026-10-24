"""Por que a leitura de pares falha no perceptron do HELOC? Diagnostico em uma semente.

Hipoteses e a medida de cada uma:
  1. modelo quase linear       acerto do MLP × regressao logistica; alinhamento (CKA) da Gram das
                               ativacoes de cada bloco com a da entrada
  2. geometria sem decisao     k-vizinhos contra a predicao e AUC da margem geometrica para o acerto,
                               por fonte e bloco, contra o piso (rede nao treinada) -- as leituras de P1
  3. fonte que atrapalha       AUC de trocas e Spearman com o Shapley exato, ranking a ranking, para cada
                               fonte isolada e para a conjunta
  4. normalizacao angular      o mesmo, sem T_a (kernel linear dividido pela media da diagonal)
  5. bloco                     tudo por bloco
Referencias de leitura: o ranking pela propria margem do modelo (Δ_f = |m(i) − m(j)| − |m(i^(f)) − m(j)|),
que usa a saida em vez da geometria, e |Δx|.

Uso:
    python heloc_diagnosis.py --seed 0 --pairs 40
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from case_study import predict
from src.attributions import baseline_shapley
from src.case import BAD, GOOD, fmt

DATASET = "heloc"  # o diagnostico e da falha no HELOC
from src.data_tabular import balanced_sample, load_tabular, split, standardize
from src.mlp import ResidualMLP
from tabular_differences import area, flip_curve
from tabular_suite import extract, train

SOURCES = ("params", "activations", "grads")
VARIANTS = {"conjunta": SOURCES, "só W": ("params",), "só A": ("activations",), "só Ω": ("grads",),
            "W e A": ("params", "activations")}


def gram_rows(src: dict, keys, n: int, angular: bool, shift: float) -> np.ndarray:
    """Gram conjunta (n × n) das linhas, com ou sem T_a, com o deslocamento por J."""
    G = np.ones((n, n))
    for key in keys:
        u = src[key].reshape(n, -1).astype(np.float64)
        K = u @ u.T
        if angular:
            d = np.sqrt(np.clip(np.diag(K), 1e-300, None))
            K = K / np.outer(d, d)
        else:
            K = K / max(np.mean(np.diag(K)), 1e-300)
        G *= (K + shift) / (1.0 + shift)
    return G


def dist_from_row0(G: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(G[0, 0] + np.diag(G) - 2 * G[0], 0.0))


def pair_variant_scores(model, xi, xj, device, target, shift) -> dict:
    """Δd_f para cada variante (fontes × T_a) e bloco."""
    F = len(xi)
    swapped = np.repeat(xi[None], F, axis=0)
    swapped[np.arange(F), np.arange(F)] = xj
    rows = np.concatenate([xj[None], xi[None], swapped]).astype(np.float32)
    sources = extract(model, rows, device, target)["sources"]
    out = {}
    for b, src in sources.items():
        for name, keys in VARIANTS.items():
            for angular in (True, False):
                d = dist_from_row0(gram_rows(src, keys, len(rows), angular, shift))
                out[(b, name, angular)] = d[1] - d[2:]
    return out


def knn_and_margin(src, keys, n, preds, correct, k=5, shift=1.0, multiclass=False) -> tuple[float, float]:
    """k-vizinhos leave-one-out contra a predicao e AUC da margem geometrica para o acerto (P1).

    A margem e a da Equacao `eq:margem_geometrica`: distancia media aos objetos da propria classe
    predita contra a menor distancia media as demais classes preditas (com duas classes, a outra).
    """
    from sklearn.metrics import roc_auc_score
    G = gram_rows(src, keys, n, True, shift)
    dg = np.diag(G)
    D = np.sqrt(np.maximum(dg[:, None] + dg[None] - 2 * G, 0.0))
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, 1)[:, :k]
    K = int(preds.max() + 1)
    vote = np.array([np.bincount(preds[row], minlength=K).argmax() for row in nn])
    knn = float((vote == preds).mean())
    np.fill_diagonal(D, np.nan)
    classes = np.unique(preds)
    means = np.stack([np.nanmean(np.where(preds[None] == c, D, np.nan), 1) for c in classes], 1)
    pos = np.searchsorted(classes, preds)
    own = means[np.arange(n), pos]
    means[np.arange(n), pos] = np.inf
    oth = means.min(1)
    mu = np.nan_to_num((oth - own) / np.maximum(oth + own, 1e-300))  # distancias todas nulas: margem 0
    return knn, float(roc_auc_score(correct, mu)) if 0 < correct.mean() < 1 else float("nan")


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y) ** 2
    return float(hsic / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--fixed-target", type=int, default=0)
    ap.add_argument("--out", default="outputs/heloc_diagnosis")
    args = ap.parse_args()
    seed = args.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- o mesmo modelo e os mesmos pares de `case_study.run_one` ---
    X, y = load_tabular(DATASET)
    tr, va, te = split(y, seed)
    X_tr, X_va, X_te = (a.astype(np.float32) for a in standardize(X[tr], X[va], X[te]))
    torch.manual_seed(seed)
    model = ResidualMLP(X.shape[1], 2, args.width, args.blocks).to(device)
    untrained = copy.deepcopy(model)
    train(model, X_tr, y[tr], X_va, y[va], device, args, seed)
    model.eval()
    untrained.eval()
    preds = predict(model, X_te, device)
    y_te, F = y[te], X.shape[1]
    rng = np.random.default_rng(seed)
    good = np.flatnonzero(preds == GOOD)
    pairs = [(int(i), int(good[np.argmin(np.linalg.norm(X_te[good] - X_te[i], axis=1))]))
             for i in rng.choice(np.flatnonzero((preds == BAD) & (y_te == BAD)), size=100, replace=False)][: args.pairs]

    L = [f"# Diagnóstico da leitura de pares no HELOC (semente {seed}, {len(pairs)} pares)\n"]

    # 1. linearidade
    logreg = LogisticRegression(max_iter=2000).fit(X_tr, y[tr])
    acc_lr = (logreg.predict(X_te) == y_te).mean()
    agree = (logreg.predict(X_te) == preds).mean()
    sample = balanced_sample(y_te, total=500, seed=seed)
    ext = extract(model, X_te[sample], device, args.fixed_target)
    ext_u = extract(untrained, X_te[sample], device, args.fixed_target)
    blocks = model.block_names
    L += ["## 1. O modelo é quase linear?\n",
          f"- acerto: MLP {fmt((preds == y_te).mean())}, regressão logística {fmt(acc_lr)}; as duas concordam em {agree:.0%} do teste",
          "- CKA linear entre as ativações de cada bloco e a entrada (treinada × não treinada): " + ", ".join(
              f"{b} {fmt(linear_cka(ext['sources'][b]['activations'].reshape(len(sample), -1), X_te[sample]), 2)}"
              f"/{fmt(linear_cka(ext_u['sources'][b]['activations'].reshape(len(sample), -1), X_te[sample]), 2)}"
              for b in blocks), ""]

    # 2. P1 no HELOC
    p_s, correct = preds[sample], (preds[sample] == y_te[sample]).astype(int)
    L += ["## 2. A geometria está organizada pela decisão? (leituras de P1 na amostra balanceada)\n",
          "k-vizinhos contra a predição / AUC da margem geométrica para o acerto; entre parênteses, a rede não treinada.\n",
          "| fonte | " + " | ".join(blocks) + " |", "|---|" + "---|" * len(blocks)]
    for name, keys in VARIANTS.items():
        cells = []
        for b in blocks:
            src, src_u = ext["sources"][b], ext_u["sources"][b]
            if any(np.median(np.linalg.norm(src[k].reshape(len(sample), -1), axis=1)) < 1e-12 for k in keys):
                cells.append("—")
                continue
            kn, au = knn_and_margin(src, keys, len(sample), p_s, correct, shift=args.shift)
            kn_u, au_u = knn_and_margin(src_u, keys, len(sample), p_s, correct, shift=args.shift)
            cells.append(f"{fmt(kn, 2)}/{fmt(au, 2)} ({fmt(kn_u, 2)}/{fmt(au_u, 2)})")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    logit_margin = ext["logits"][:, 1] - ext["logits"][:, 0]
    from sklearn.metrics import roc_auc_score
    L += ["", f"Referência: AUC da |margem dos logits| para o acerto {fmt(roc_auc_score(correct, np.abs(logit_margin)), 2)}.\n"]

    # 3-5. leitura de pares por variante
    grid = np.linspace(0, 1, F + 1)
    curves, spear_shap, spear_abs = {}, {}, {}
    for n, (i, j) in enumerate(pairs):
        xi, xj = X_te[i], X_te[j]
        sc = pair_variant_scores(model, xi, xj, device, args.fixed_target, args.shift)
        phi, _ = baseline_shapley(model, xi, xj, BAD, GOOD, device)
        with torch.no_grad():
            Fr = np.repeat(xi[None], F, axis=0)
            Fr[np.arange(F), np.arange(F)] = xj
            z = model(torch.from_numpy(np.concatenate([xj[None], xi[None], Fr])).to(device)).double().cpu().numpy()
        m = z[:, GOOD] - z[:, BAD]
        sc[("saída", "margem do modelo", True)] = np.abs(m[1] - m[0]) - np.abs(m[2:] - m[0])
        sc[("—", "|Δx|", True)] = np.abs(xj - xi)
        sc[("—", "Shapley exato", True)] = phi
        differ = np.flatnonzero(np.abs(xj - xi) > 1e-9)
        for key, s in sc.items():
            order = np.argsort(-s, kind="stable")
            curves.setdefault(key, []).append(flip_curve(model, xi, xj, GOOD, order, grid, device))
            if len(differ) >= 3:
                spear_shap.setdefault(key, []).append(spearmanr(s[differ], phi[differ])[0])
                spear_abs.setdefault(key, []).append(spearmanr(s[differ], np.abs(xj - xi)[differ])[0])
        print(f"par {n + 1}/{len(pairs)}", flush=True)

    auc = {k: area(np.mean(v, 0), grid) for k, v in curves.items()}
    L += ["## 3-5. Leitura de pares por fonte, normalização e bloco\n",
          "AUC de trocas / Spearman mediano com o Shapley exato / Spearman mediano com |Δx| (só atributos que diferem).\n",
          "| ranking | T_a | " + " | ".join(blocks) + " |", "|---|---|" + "---|" * len(blocks)]
    for name in VARIANTS:
        for angular in (True, False):
            cells = [f"{fmt(auc[(b, name, angular)], 2)} / {fmt(np.nanmedian(spear_shap[(b, name, angular)]), 2)} / "
                     f"{fmt(np.nanmedian(spear_abs[(b, name, angular)]), 2)}" if (b, name, angular) in auc else "—"
                     for b in blocks]
            L.append(f"| {name} | {'sim' if angular else 'não'} | " + " | ".join(cells) + " |")
    L += ["", "Referências:\n"]
    for key in (("saída", "margem do modelo", True), ("—", "|Δx|", True), ("—", "Shapley exato", True)):
        L.append(f"- {key[1]}: AUC {fmt(auc[key], 3)}, Spearman com o Shapley {fmt(np.nanmedian(spear_shap[key]), 2)}, "
                 f"com |Δx| {fmt(np.nanmedian(spear_abs[key]), 2)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "auc.json").write_text(json.dumps({" | ".join(map(str, k)): v for k, v in auc.items()}), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

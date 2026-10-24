"""Contrafactual sequencial guiado pela geometria, na arvore de decisao (base de `FANNAR_CASE`).

O teste de trocas de `tree_case_study.py` leva os atributos de i ao valor de j um de cada vez, na
ordem de Δd_f calculada uma unica vez a partir de i, e por isso nao enxerga interacoes: o metodo
precisa de ~5,5 trocas onde o minimo e ~1,8. Aqui o contrafactual e construido passo a passo:

  alvo        S = os k objetos preditos como c_j mais proximos de x_i na Gram conjunta (profundidade
              maxima), e μ = media de S nos atributos. Interpolar em direcao a μ, e nao a um unico
              j, permite combinar caracteristicas de varias amostras da outra classe;
  ranking     em cada passo, Δ_f = D(x) − D(x com x_f := μ_f), com D(x) a distancia media de x aos
              objetos de S na Gram conjunta;
  passo       o atributo escolhido anda de x_f a μ_f em fracoes t = 1/4, 2/4, 3/4, 1; a primeira
              fracao que muda a predicao para c_j encerra o contrafactual. Se nenhuma muda, o
              atributo fica em μ_f, o ranking e recalculado a partir do novo ponto e segue-se.

Variantes, para separar o que vem de cada escolha:

  sequencial (geometria)   o procedimento acima;
  ranking fixo (geometria) a mesma ordem de Δ_f calculada so a partir de x_i, sem recalcular;
  sequencial (saida)       o mesmo procedimento, mas escolhendo o atributo pelo ganho em
                           P(c_j) − P(c_i) da arvore -- a guia pela saida, como no NICE;
  aleatoria                ordem aleatoria entre os atributos que diferem de μ;
  minimo exato             menor subconjunto de atributos levados a μ que muda a predicao
                           (`tree_min_flip`, exato como a enumeracao dos 2^F subconjuntos, t = 1).

Pares: os de `src.case.select_pairs`, sobre as predicoes da propria arvore. Cada procedimento roda com
todos os atributos e so com os mutaveis (`src.case.IMMUTABLE` congelados).

Medidas por par: validade, numero de atributos alterados, distancia L1 padronizada a x_i e
plausibilidade (distancia padronizada ao objeto de treino mais proximo predito como c_j).

Uso:
    python tree_counterfactual.py --seeds 5 --pairs 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.case import DATASET, IMMUTABLE, MIN_LEAF, mutable_indices, order_of, pm, select_pairs, tree_min_flip
from src.data_tabular import feature_names, load_tabular, split
from tree_case_study import TreeGeometry, flip, method_scores, tree_shapley

STEPS = (0.25, 0.5, 0.75, 1.0)
VARIANTS = ("seq_geo", "fixed_geo", "seq_out", "random")
LABEL = {"seq_geo": "sequencial, geometria", "fixed_geo": "ranking fixo, geometria",
         "seq_out": "sequencial, saída da árvore", "random": "aleatória",
         "minimal": "mínimo exato (enumeração)"}


def mean_distance(geo: TreeGeometry, X: np.ndarray, S: np.ndarray, depth: int) -> np.ndarray:
    """D(x) para cada linha de X: distancia media aos objetos de S na Gram conjunta."""
    out = np.zeros(len(X))
    for s in S:
        out += geo.distances(np.concatenate([s[None], X]), depth)[1:]
    return out / len(S)


def margin(tree, X: np.ndarray, ci: int, cj: int) -> np.ndarray:
    col = {c: k for k, c in enumerate(tree.classes_)}
    p = tree.predict_proba(X.astype(np.float32))
    return p[:, col[cj]] - p[:, col[ci]]


def counterfactual(tree, geo, xi, mu, S, ci, cj, depth, mode: str, rng, allowed):
    """Constroi o contrafactual; devolve (x', atributos alterados em ordem, valido)."""
    x = xi.copy()
    remaining = [int(f) for f in allowed if abs(mu[f] - xi[f]) > 1e-9]
    if mode == "fixed_geo":
        cand = np.repeat(xi[None], len(remaining), axis=0)
        cand[np.arange(len(remaining)), remaining] = mu[remaining]
        gain = mean_distance(geo, xi[None], S, depth)[0] - mean_distance(geo, cand, S, depth)
        queue = [remaining[k] for k in order_of(gain, rng)]
    elif mode == "random":
        queue = list(rng.permutation(remaining))
    changed = []
    while remaining:
        if mode in ("seq_geo", "seq_out"):
            cand = np.repeat(x[None], len(remaining), axis=0)
            cand[np.arange(len(remaining)), remaining] = mu[remaining]
            if mode == "seq_geo":
                gain = mean_distance(geo, x[None], S, depth)[0] - mean_distance(geo, cand, S, depth)
            else:
                gain = margin(tree, cand, ci, cj) - margin(tree, x[None], ci, cj)[0]
            f = remaining[int(order_of(gain, rng)[0])]
        else:
            f = queue.pop(0)
        remaining.remove(f)
        changed.append(int(f))
        start = x[f]
        for t in STEPS:
            x[f] = start + t * (mu[f] - start)
            if tree.predict(x[None].astype(np.float32))[0] == cj:
                return x, changed, True
    return x, changed, False


def run_one(seed: int, args) -> dict:
    X, y = load_tabular(DATASET)
    tr, va, te = split(y, seed)
    raw_tr, raw_te, y_te = X[tr].astype(np.float64), X[te].astype(np.float64), y[te]
    rng = np.random.default_rng(seed)
    tree = DecisionTreeClassifier(min_samples_leaf=args.min_leaf, random_state=seed).fit(raw_tr, y[tr])
    mutable = mutable_indices(feature_names(DATASET))
    scale = np.maximum(raw_tr.std(0), 1e-6)
    geo = TreeGeometry(tree, scale)
    depth = geo.max_depth
    preds, preds_tr = tree.predict(raw_te), tree.predict(raw_tr)
    Xs_te = (raw_te - raw_tr.mean(0)) / scale

    records = []
    for i, j in select_pairs(preds, y_te, Xs_te, rng, args.pairs):
        xi, xj, ci, cj = raw_te[i], raw_te[j], int(preds[i]), int(preds[j])

        # o teste atual: trocas em direcao a j, na ordem de Δ_f calculada a partir de i
        c = flip(tree, xi, xj, cj, order_of(method_scores(geo, xi, xj, [depth])[depth], rng))
        k_swap = int(np.flatnonzero(c > 0.5)[0])
        k_min_j = tree_shapley(tree, xi, xj, ci, cj)[2]

        # alvo: os k objetos preditos como c_j mais proximos de i na geometria do metodo
        same = np.flatnonzero(preds == cj)
        d = geo.distances(np.concatenate([xi[None], raw_te[same]]), depth)[1:]
        S = raw_te[same[np.argsort(d)[: args.k]]]
        mu = S.mean(0)

        rec = {"i": int(i), "j": j, "pred_i": ci, "pred_j": cj, "k_swap_j": k_swap, "k_min_j": k_min_j,
               "mu_is_cj": bool(tree.predict(mu[None].astype(np.float32))[0] == cj)}
        target_tr = raw_tr[preds_tr == cj]
        for scope, allowed in (("all", np.arange(len(xi))),) + ((("mutable", mutable),) if IMMUTABLE else ()):
            res = {"k_min_mu": tree_min_flip(tree, xi, mu, cj, allowed), "variants": {}}
            for v in VARIANTS:
                t0 = time.perf_counter()
                x, changed, valid = counterfactual(tree, geo, xi, mu, S, ci, cj, depth, v, rng, allowed)
                res["variants"][v] = {
                    "valid": valid, "n_changed": len(changed), "changed": changed,
                    "l1": float(np.abs((x - xi) / scale).sum()),
                    "plaus": float(np.min(np.linalg.norm((target_tr - x) / scale, axis=1))),
                    "seconds": time.perf_counter() - t0}
            rec[scope] = res
        records.append(rec)
    return {"seed": seed, "accuracy": float((preds == y_te).mean()), "depth": depth, "k": args.k,
            "pairs": records}


def write_report(runs: list[dict], args, out: Path) -> None:
    def per_seed(fn):
        return [fn(r["pairs"]) for r in runs]

    L = [f"# Contrafactual sequencial guiado pela geometria — árvore, `{DATASET}`\n",
         f"{len(runs)} sementes × {args.pairs} pares. Alvo: média dos {args.k} objetos preditos na classe de j "
         "mais próximos de i na Gram conjunta (profundidade máxima). Média ± desvio entre sementes.\n",
         f"- a média μ já é classificada na classe de j em {pm(per_seed(lambda P: np.mean([p['mu_is_cj'] for p in P])))} dos pares",
         f"- teste atual (trocas em direção a j, ordem de Δ_f a partir de i): "
         f"{pm(per_seed(lambda P: np.mean([p['k_swap_j'] for p in P])), 2)} trocas; mínimo exato em direção a j "
         f"{pm(per_seed(lambda P: np.mean([p['k_min_j'] for p in P if p['k_min_j'] is not None])), 2)}\n"]
    for scope in [s for s in ("all", "mutable") if s in runs[0]["pairs"][0]]:
        L += [f"## Escopo: {'todos os atributos' if scope == 'all' else 'só os mutáveis'}\n",
              "| procedimento | válido | atributos alterados | excesso sobre o mínimo | atinge o mínimo | L1 padronizada | "
              "plausibilidade | ms por par |", "|---|---|---|---|---|---|---|---|"]
        for v in VARIANTS:
            def stat(key, P, v=v):
                return np.mean([p[scope]["variants"][v][key] for p in P if p[scope]["variants"][v]["valid"]])

            def ok(P, v=v):
                return [p for p in P if p[scope]["variants"][v]["valid"] and p[scope]["k_min_mu"] is not None]

            def excess(P, v=v):
                return np.mean([p[scope]["variants"][v]["n_changed"] - p[scope]["k_min_mu"] for p in ok(P)])

            def hits(P, v=v):
                return np.mean([p[scope]["variants"][v]["n_changed"] == p[scope]["k_min_mu"] for p in ok(P)])
            L.append(f"| {LABEL[v]} | {pm(per_seed(lambda P: np.mean([p[scope]['variants'][v]['valid'] for p in P])))} | "
                     f"{pm(per_seed(lambda P: stat('n_changed', P)), 2)} | {pm(per_seed(excess), 2)} | {pm(per_seed(hits))} | "
                     f"{pm(per_seed(lambda P: stat('l1', P)), 2)} | {pm(per_seed(lambda P: stat('plaus', P)), 2)} | "
                     f"{pm(per_seed(lambda P: stat('seconds', P) * 1e3), 1)} |")
        L.append(f"| {LABEL['minimal']} | {pm(per_seed(lambda P: np.mean([p[scope]['k_min_mu'] is not None for p in P])))} | "
                 f"{pm(per_seed(lambda P: np.mean([p[scope]['k_min_mu'] for p in P if p[scope]['k_min_mu'] is not None])), 2)} "
                 "| 0 | 100% | — | — | — |\n")
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--k", type=int, default=5, help="objetos da outra classe cuja media e o alvo")
    ap.add_argument("--min-leaf", type=int, default=MIN_LEAF, help="o de `src.case.MIN_LEAF`, escolhido pela validacao")
    ap.add_argument("--out", default=f"outputs/tree_counterfactual_{DATASET}")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in range(args.seeds):
        path = out / "runs" / f"seed{seed}.json"
        if path.exists():
            runs.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        t0 = time.perf_counter()
        r = run_one(seed, args)
        path.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        runs.append(r)
        ch = {v: np.mean([p["all"]["variants"][v]["n_changed"] for p in r["pairs"] if p["all"]["variants"][v]["valid"]])
              for v in VARIANTS}
        print(f"semente {seed} ({time.perf_counter() - t0:.0f} s): mínimo "
              f"{np.mean([p['all']['k_min_mu'] for p in r['pairs'] if p['all']['k_min_mu'] is not None]):.2f} | "
              + " ".join(f"{v} {c:.2f}" for v, c in ch.items()), flush=True)
    write_report(runs, args, out)
    print(f"pronto: {out}/")


if __name__ == "__main__":
    main()

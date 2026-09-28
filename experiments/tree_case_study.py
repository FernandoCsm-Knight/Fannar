"""O metodo aplicado a uma arvore de decisao: ele diz o que a propria arvore explica?

O metodo e a forma (fontes por objeto, nucleo por fonte, transformacao, produto de Hadamard);
parametros, ativacoes e gradiente sao so a instanciacao usada nas redes. Aqui a instanciacao
e outra, sem nada de rede. Os "blocos" sao as profundidades ℓ da arvore, e cada amostra tem,
em cada ℓ, tres fontes:

  caminho   indicador dos nos visitados ate a profundidade ℓ (o estado da arvore);
  folga     (x_f − t)/σ_f em cada no interno ja atravessado (profundidade < ℓ): quanto a
            amostra passou de cada limiar que usou, isto e, quanto falta para mudar de ramo;
  classes   distribuicao de classes do no em que a amostra esta na profundidade ℓ.

A composicao e a das redes: k_l = (cos + 1)/2 por fonte, produto de Hadamard, d² = 2(1 − Π k_l),
e a leitura de pares e a mesma Δd_f = d(i, j) − d(i^(f), j).

A arvore tem explicacao propria, e o que se testa e se o metodo a recupera:

  1. corte      o atributo do no em que os caminhos de i e j se separam -- e o 1º do metodo?
  2. datacao    a profundidade em que esse atributo passa a ser o 1º do metodo coincide com a
                profundidade em que os caminhos se separam?
  3. troca      teste contrafactual na propria arvore, e o numero de trocas que cada ranking
                precisa contra o minimo exato (enumeracao dos 2^F subconjuntos);
  4. global     parcela media do metodo contra a importancia de Gini da arvore.

Referencias: Shapley exato da arvore com j de referencia (jogo sobre P(c_j) − P(c_i)), SHAP de
arvore (TreeExplainer) para o global, |Δx|, ordem aleatoria entre os atributos que diferem e o
piso -- o metodo numa arvore treinada com os rotulos embaralhados (mesmos hiperparametros).
Empates no ranking sao desfeitos ao acaso, em todos os rankings.

Uso:
    python tree_case_study.py --seeds 5 --pairs 100
"""

from __future__ import annotations

import argparse
import json
import time
from math import factorial
from pathlib import Path

import numpy as np
import shap
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeClassifier

from letter_case_study import DATASET, DESCRIPTION, fmt, normalized, pm, rank_of, split_node
from src.data_tabular import class_names, feature_names, load_tabular, split, standardize
from tabular_differences import area

SOURCES = ("caminho", "folga", "classes")
RANKINGS = ("method", "shapley", "absdiff", "random", "floor")
LABEL = {"method": "método (Δd na Gram conjunta)", "shapley": "Shapley exato com j de referência",
         "absdiff": "|Δx|", "random": "aleatória entre os que diferem",
         "floor": "método na árvore de rótulos embaralhados",
         "caminho": "só caminho", "folga": "só folga", "classes": "só classes"}


class TreeGeometry:
    """As tres fontes de uma arvore por profundidade, e as distancias da Gram conjunta."""

    def __init__(self, tree: DecisionTreeClassifier, scale: np.ndarray) -> None:
        t = tree.tree_
        self.tree, self.scale = tree, scale
        self.internal = t.children_left != -1
        self.feature = np.where(self.internal, t.feature, 0)
        self.threshold = t.threshold
        self.value = t.value[:, 0, :] / t.value[:, 0, :].sum(1, keepdims=True)
        self.depth = np.zeros(t.node_count, dtype=int)
        for n in range(t.node_count):  # pre-ordem: o pai vem antes do filho
            for child in (t.children_left[n], t.children_right[n]):
                if child != -1:
                    self.depth[child] = self.depth[n] + 1
        self.max_depth = int(self.depth.max())

    def sources(self, X: np.ndarray, depth: int) -> dict:
        P = self.tree.decision_path(X.astype(np.float32)).toarray().astype(np.float64)
        path = P * (self.depth <= depth)
        crossed = P * (self.depth < depth) * self.internal
        slack = crossed * (X[:, self.feature] - self.threshold) / self.scale[self.feature]
        node = (path * np.arange(path.shape[1])).argmax(1)  # o mais profundo do prefixo
        return {"caminho": path, "folga": slack, "classes": self.value[node]}

    def distances(self, X: np.ndarray, depth: int, only: tuple[str, ...] = SOURCES) -> np.ndarray:
        """d(linha 0, cada linha), como em `tabular_differences.joint_distances`."""
        joint = np.ones(len(X))
        for key, u in self.sources(X, depth).items():
            if key not in only:
                continue
            norms = np.linalg.norm(u, axis=1)
            both_zero = (norms < 1e-12) & (norms[0] < 1e-12)
            cos = np.where(both_zero, 1.0, (u @ u[0]) / np.maximum(norms * norms[0], 1e-300))
            joint *= (cos + 1.0) / 2.0
        return np.sqrt(np.maximum(2.0 * (1.0 - joint), 0.0))


def swap_rows(xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
    F = len(xi)
    swapped = np.repeat(xi[None], F, axis=0)
    swapped[np.arange(F), np.arange(F)] = xj
    return np.concatenate([xj[None], xi[None], swapped])


def method_scores(geo: TreeGeometry, xi, xj, depths, only=SOURCES) -> dict:
    rows = swap_rows(xi, xj)
    out = {}
    for ell in depths:
        d = geo.distances(rows, ell, only)
        out[ell] = d[1] - d[2:]
    return out


def tree_shapley(tree, xi, xj, ci: int, cj: int):
    """Shapley exato de v(S) = [P(c_j) − P(c_i)](x_j em S, x_i fora) e o menor S que muda a predicao."""
    F = len(xi)
    masks = np.arange(1 << F)
    bits = ((masks[:, None] >> np.arange(F)) & 1).astype(bool)
    rows = np.where(bits, xj, xi).astype(np.float32)
    proba = tree.predict_proba(rows)
    col = {c: k for k, c in enumerate(tree.classes_)}
    v = proba[:, col[cj]] - proba[:, col[ci]]
    size = bits.sum(1)
    w = np.array([factorial(k) * factorial(F - k - 1) / factorial(F) for k in range(F)])
    phi = np.empty(F)
    for f in range(F):
        without = masks[~bits[:, f]]
        phi[f] = (w[size[without]] * (v[without | (1 << f)] - v[without])).sum()
    flipped = tree.classes_[proba.argmax(1)] == cj
    return phi, float(v[-1] - v[0]), int(size[flipped].min())


def order_of(score: np.ndarray, rng) -> np.ndarray:
    """Decrescente, com empates desfeitos ao acaso (lexsort ordena pela ultima chave)."""
    return np.lexsort((rng.random(len(score)), -np.asarray(score)))


def flip(tree, xi, xj, cj: int, order: np.ndarray) -> np.ndarray:
    """Para k = 0..F: i com os k primeiros atributos de j e classificada como j?"""
    F = len(xi)
    rank = np.empty(F, dtype=int)
    rank[order] = np.arange(F)
    batch = np.repeat(xi[None], F + 1, axis=0)
    for k in range(F + 1):
        batch[k, rank < k] = xj[rank < k]
    return (tree.predict(batch.astype(np.float32)) == cj).astype(float)


def run_one(seed: int, args) -> dict:
    X, y = load_tabular(DATASET)
    tr, va, te = split(y, seed)
    Xs_tr, _, Xs_te = standardize(X[tr], X[va], X[te])
    raw_tr, raw_te, y_te = X[tr], X[te], y[te]
    rng = np.random.default_rng(seed)
    tree = DecisionTreeClassifier(min_samples_leaf=args.min_leaf, random_state=seed).fit(raw_tr, y[tr])
    shuffled = DecisionTreeClassifier(min_samples_leaf=args.min_leaf, random_state=seed).fit(
        raw_tr, rng.permutation(y[tr]))
    scale = np.maximum(raw_tr.std(0), 1e-6)
    geo, geo_u = TreeGeometry(tree, scale), TreeGeometry(shuffled, scale)
    preds = tree.predict(raw_te)
    F = X.shape[1]
    depths = list(range(1, geo.max_depth + 1))
    full = geo.max_depth

    records, curves = [], {k: [] for k in (*RANKINGS, *SOURCES)}
    t_method, t_shap = [], []
    for i in rng.choice(np.flatnonzero(preds == y_te), size=args.pairs, replace=False):
        other = np.flatnonzero(preds != preds[i])
        j = int(other[np.argmin(np.linalg.norm(Xs_te[other] - Xs_te[i], axis=1))])
        xi, xj, ci, cj = raw_te[i].astype(np.float64), raw_te[j].astype(np.float64), int(preds[i]), int(preds[j])
        differ = np.flatnonzero(np.abs(xj - xi) > 1e-9)

        t0 = time.perf_counter()
        scores = method_scores(geo, xi, xj, depths)
        t_method.append(time.perf_counter() - t0)
        floor = method_scores(geo_u, xi, xj, [geo_u.max_depth])[geo_u.max_depth]
        single = {s: method_scores(geo, xi, xj, [full], (s,))[full] for s in SOURCES}
        t0 = time.perf_counter()
        phi, total, k_min = tree_shapley(tree, xi, xj, ci, cj)
        t_shap.append(time.perf_counter() - t0)
        assert abs(phi.sum() - total) < 1e-9, "eficiencia do Shapley exato"

        random_score = np.zeros(F)
        random_score[differ] = rng.random(len(differ)) + 1.0
        orders = {"method": order_of(scores[full], rng), "shapley": order_of(phi, rng),
                  "absdiff": order_of(np.abs(xj - xi), rng), "random": order_of(random_score, rng),
                  "floor": order_of(floor, rng)} | {s: order_of(single[s], rng) for s in SOURCES}
        k_needed = {}
        for k, o in orders.items():
            c = flip(tree, xi, xj, cj, o)
            curves[k].append(c)
            k_needed[k] = int(np.flatnonzero(c > 0.5)[0])

        node = split_node(tree, xi, xj)
        f_cut = int(tree.tree_.feature[node])
        cut_depth = int(geo.depth[node])  # o no testa f na profundidade dele; os filhos ja diferem
        # o corte "vira o 1º" quando o seu Δd e positivo e maximo; sem isso, profundidades em que
        # todo Δd e zero o poriam em 1º pelo desempate aleatorio
        first_top = next((ell for ell in depths
                          if scores[ell][f_cut] > 1e-12 and scores[ell][f_cut] >= scores[ell].max()), None)
        records.append({
            "i": int(i), "j": j, "pred_i": ci, "pred_j": cj, "n_differ": int(len(differ)),
            "raw_i": xi.tolist(), "raw_j": xj.tolist(), "cut_feature": f_cut, "cut_depth": cut_depth,
            "cut_threshold": float(tree.tree_.threshold[node]), "leaf_depth_i": int(geo.depth[tree.apply(xi[None].astype(np.float32))[0]]),
            "first_top_depth": first_top, "k_min": k_min, "k_needed": k_needed,
            "method_full": scores[full].tolist(), "shapley": phi.tolist(),
            "method_by_depth": {str(ell): scores[ell].tolist() for ell in depths},
            "positions": {k: rank_of(o, f_cut) for k, o in orders.items()},
        })

    grid = np.arange(F + 1) / F
    mean_curves = {k: np.mean(v, 0).tolist() for k, v in curves.items()}
    tree_shap = shap.TreeExplainer(tree).shap_values(raw_te[: args.global_samples].astype(np.float64))
    tree_shap = np.asarray(tree_shap)
    if tree_shap.shape[0] != args.global_samples:  # (F, n, K) ou (n, F, K) conforme a versao
        tree_shap = np.moveaxis(tree_shap, 0, -1)
    p = preds[: args.global_samples]
    shap_global = np.abs(tree_shap[np.arange(len(p)), :, p]).mean(0)
    return {
        "seed": seed, "accuracy": float((preds == y_te).mean()), "depth": geo.max_depth,
        "leaves": int(tree.get_n_leaves()), "shuffled_accuracy": float((shuffled.predict(raw_te) == y_te).mean()),
        "grid": grid.tolist(), "curves": mean_curves,
        "auc": {k: area(np.asarray(v), grid) for k, v in mean_curves.items()},
        "pairs": records,
        "global": {"method": np.mean([normalized(r["method_full"]) for r in records], 0).tolist(),
                   "gini": tree.feature_importances_.tolist(), "tree_shap": shap_global.tolist()},
        "cost": {"method_s": float(np.mean(t_method)), "shapley_s": float(np.mean(t_shap)), "depths": len(depths)},
    }


def write_report(runs: list[dict], args, out: Path) -> None:
    feats, classes = feature_names(DATASET), class_names(DATASET)
    pairs = [p for r in runs for p in r["pairs"]]
    L = ["# O método aplicado a uma árvore de decisão — `letter`\n",
         f"Árvore treinada nos rótulos (min_samples_leaf = {args.min_leaf}); {len(runs)} sementes × {args.pairs} pares "
         f"(i acertada pela árvore, j a mais próxima com outra predição da árvore). Método na profundidade máxima, "
         "salvo menção. Média ± desvio entre sementes.\n",
         f"- árvore: acerto {pm([r['accuracy'] for r in runs])}, {pm([r['leaves'] for r in runs], 0)} folhas, "
         f"profundidade {pm([r['depth'] for r in runs], 0)}",
         f"- árvore de rótulos embaralhados (piso): acerto {pm([r['shuffled_accuracy'] for r in runs])}",
         f"- atributos que diferem entre i e j: {pm([p['n_differ'] for p in pairs], 1)} de 16\n"]

    L += ["## 1. O atributo em que a árvore separa i de j\n",
          "| ranking | 1º do ranking | entre os 3 primeiros | posto recíproco médio |", "|---|---|---|---|"]
    for k in (*RANKINGS, *SOURCES):
        per = [[p["positions"][k] for p in r["pairs"]] for r in runs]
        L.append(f"| {LABEL[k]} | {pm([np.mean(np.asarray(v) == 0) for v in per])} | "
                 f"{pm([np.mean(np.asarray(v) < 3) for v in per])} | {pm([np.mean(1 / (np.asarray(v) + 1)) for v in per])} |")
    L.append(f"| acaso entre os que diferem | {pm([np.mean([1 / p['n_differ'] for p in r['pairs']]) for r in runs])} | "
             f"{pm([np.mean([min(3, p['n_differ']) / p['n_differ'] for p in r['pairs']]) for r in runs])} | — |")

    diff = np.array([p["first_top_depth"] - (p["cut_depth"] + 1) for p in pairs if p["first_top_depth"] is not None])
    never = np.mean([p["first_top_depth"] is None for p in pairs])
    cut = np.array([p["cut_depth"] + 1 for p in pairs if p["first_top_depth"] is not None])
    first = np.array([p["first_top_depth"] for p in pairs if p["first_top_depth"] is not None])
    L += ["", "## 2. Datação: em que profundidade o atributo do corte vira o 1º do método\n",
          "Comparada à profundidade em que os caminhos de i e j se separam (a do corte + 1).\n",
          f"- exatamente na profundidade da separação: {np.mean(diff == 0):.0%} dos pares",
          f"- antes: {np.mean(diff < 0):.0%}; depois: {np.mean(diff > 0):.0%}; nunca vira o 1º: {never:.0%}",
          f"- Spearman entre as duas profundidades: {fmt(spearmanr(first, cut)[0])} "
          f"(profundidade da separação: mediana {np.median(cut):.0f}, de {cut.min()} a {cut.max()})\n"]

    L += ["## 3. Teste contrafactual na própria árvore\n",
          "AUC da fração de pares classificados como j conforme os k primeiros atributos são trocados, e o número "
          "de trocas até a predição mudar, contra o mínimo exato (enumeração dos 2^16 subconjuntos).\n",
          "| ranking | AUC | trocas até mudar | excesso sobre o mínimo | atinge o mínimo |", "|---|---|---|---|---|"]
    for k in (*RANKINGS, *SOURCES):
        L.append(f"| {LABEL[k]} | {pm([r['auc'][k] for r in runs])} | "
                 f"{pm([np.mean([p['k_needed'][k] for p in r['pairs']]) for r in runs], 2)} | "
                 f"{pm([np.mean([p['k_needed'][k] - p['k_min'] for p in r['pairs']]) for r in runs], 2)} | "
                 f"{pm([np.mean([p['k_needed'][k] == p['k_min'] for p in r['pairs']]) for r in runs])} |")
    L.append(f"| mínimo exato | — | {pm([np.mean([p['k_min'] for p in r['pairs']]) for r in runs], 2)} | 0 | 100% |")
    sp = [spearmanr(np.asarray(p["method_full"]), np.asarray(p["shapley"]))[0] for p in pairs]
    L += ["", f"Método × Shapley exato, par a par: Spearman mediano {fmt(np.nanmedian(sp))}; mesmo 1º atributo em "
          f"{np.mean([np.argmax(p['method_full']) == np.argmax(p['shapley']) for p in pairs]):.0%} dos pares.",
          f"Custo por par: método em todas as {runs[0]['cost']['depths']} profundidades "
          f"{fmt(np.mean([r['cost']['method_s'] for r in runs]) * 1e3, 1)} ms; Shapley exato "
          f"{fmt(np.mean([r['cost']['shapley_s'] for r in runs]) * 1e3, 1)} ms.\n"]

    L += ["## 4. Importância global\n", "| comparação | Spearman |", "|---|---|",
          f"| método × Gini da árvore | {pm([spearmanr(r['global']['method'], r['global']['gini'])[0] for r in runs])} |",
          f"| SHAP de árvore × Gini da árvore | {pm([spearmanr(r['global']['tree_shap'], r['global']['gini'])[0] for r in runs])} |",
          f"| método × SHAP de árvore | {pm([spearmanr(r['global']['method'], r['global']['tree_shap'])[0] for r in runs])} |", ""]
    gini = np.array([normalized(r["global"]["gini"]) for r in runs]).mean(0)
    L += ["| atributo | significado | Gini | método | SHAP de árvore |", "|---|---|---|---|---|"]
    meth = np.array([normalized(r["global"]["method"]) for r in runs]).mean(0)
    ts = np.array([normalized(r["global"]["tree_shap"]) for r in runs]).mean(0)
    for f in np.argsort(-gini):
        L.append(f"| {feats[f]} | {DESCRIPTION[feats[f]]} | {fmt(gini[f])} | {fmt(meth[f])} | {fmt(ts[f])} |")
    (out / "tables.md").write_text("\n".join(L) + "\n")

    E = ["# Exemplos — o método lendo a árvore (semente 0)\n"]
    for p in runs[0]["pairs"][: args.examples]:
        m = np.asarray(p["method_full"])
        mo = np.argsort(-m)
        f = p["cut_feature"]
        E += [f"## {classes[p['pred_i']]} (i) × {classes[p['pred_j']]} (j)\n",
              f"A árvore separa os dois em **{feats[f]}** ({DESCRIPTION[feats[f]]}) ≤ {fmt(p['cut_threshold'], 1)}, "
              f"na profundidade {p['cut_depth']}: i = {p['raw_i'][f]:.0f}, j = {p['raw_j'][f]:.0f}. "
              f"No método ele é o {rank_of(mo, f) + 1}º e vira o 1º na profundidade {p['first_top_depth']}. "
              f"Trocas até mudar: método {p['k_needed']['method']}, mínimo {p['k_min']}.\n",
              "| atributo | i | j | parcela no método | parcela no Shapley exato |", "|---|---|---|---|---|"]
        mp, spp = normalized(m), normalized(p["shapley"])
        for q in mo[:4]:
            E.append(f"| {feats[q]} | {p['raw_i'][q]:.0f} | {p['raw_j'][q]:.0f} | {mp[q]:.0%} | {spp[q]:.0%} |")
        E.append("")
    (out / "exemplos.md").write_text("\n".join(E) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--min-leaf", type=int, default=3)
    ap.add_argument("--global-samples", type=int, default=1000)
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--out", default="outputs/tree_case")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in range(args.seeds):
        path = out / "runs" / f"seed{seed}.json"
        if path.exists():
            runs.append(json.loads(path.read_text()))
            continue
        t0 = time.perf_counter()
        r = run_one(seed, args)
        path.write_text(json.dumps(r, ensure_ascii=False))
        runs.append(r)
        a = r["auc"]
        print(f"semente {seed} ({time.perf_counter() - t0:.0f} s): acerto {r['accuracy']:.3f} | AUC método "
              f"{a['method']:.3f} Shapley {a['shapley']:.3f} |Δx| {a['absdiff']:.3f} piso {a['floor']:.3f}", flush=True)
    write_report(runs, args, out)
    from src.plots_case import plot_tree_case
    plot_tree_case(runs, LABEL, str(out / "tree_case.png"))
    print(f"pronto: {out}/")


if __name__ == "__main__":
    main()

"""Estudo de caso de P6: por que i foi classificado assim, e o que precisaria mudar para ser classificado como j?

O perceptron multicamadas residual das suites e treinado na base de `FANNAR_CASE` (`src/case.py`): a
letter, de decisao nitida, ou o HELOC, binario e de decisao fraca. Cada par e um objeto i
classificado corretamente numa classe de que faz sentido querer sair e o objeto j mais proximo nos
atributos que o modelo classifica numa classe aceita como destino (`src.case.select_pairs`).

Tres partes.

1. **Explicar a diferenca (ranking).** Cada ranking de atributos passa pelo teste contrafactual de
   trocas (levar os k primeiros atributos de i ao valor de j). Rankings: o metodo (Δ_f na Gram
   conjunta de cada bloco), o metodo sem Ω (so W e A), o Shapley exato com j de referencia
   (2^23 subconjuntos na GPU), o SHAP da biblioteca φ(j) − φ(i) e so de i, o LIME contrastivo e so
   de i, os gradientes integrados de x_i a x_j, o DiCE (frequencia de alteracao), |Δx|, grad×Δx,
   a ordem aleatoria entre os que diferem e o metodo na rede nao treinada. Corroboracao: o atributo
   em que uma arvore (substituta, ajustada as predicoes do MLP, e nos rotulos) separa i de j.

2. **Construir o contrafactual.** Alvo: a media μ dos k objetos preditos na classe de j mais proximos
   de i na Gram conjunta do bloco. Procedimentos:
     seq_geo        a cada passo, o atributo que mais aproxima x de S na Gram conjunta anda para μ_f
                    em fracoes de 1/4 ate a predicao mudar; se nao mudar, fica em μ_f e o ranking
                    e recalculado (Algoritmo `alg:contrafactual`);
     seq_geo_noΩ    o mesmo, na Gram de W e A;
     fixed_geo      o ranking de Δ calculado so a partir de x_i;
     seq_out        o atributo escolhido pelo ganho na margem f_{c_j} − f_{c_i} (como o NICE);
     random         ordem aleatoria;
     dice           os contrafactuais do DiCE (metodo `random`) para a classe de j: o mais esparso
                    dos validos e a media sobre os validos;
     minimal        o menor subconjunto de atributos levados a μ que muda a predicao (2^23 na GPU).
   Cada um roda com todos os atributos e, se a base tiver imutaveis (`IMMUTABLE`), so com os mutaveis.
   Medidas: validade, atributos alterados, distancia L1 padronizada a x_i, plausibilidade
   (distancia padronizada ao objeto de treino mais proximo predito na classe de j) e tempo.

3. **Diagnostico de Ω.** Em cada bloco, a razao de participacao da Gram angular de cada fonte e o
   cosseno medio entre fontes de objetos distintos, na amostra balanceada do teste. Com
   `--compare`, o mesmo diagnostico numa base multiclasse da suite, para contraste.

Saidas em `outputs/<base>_case/`: `runs/seed{s}.json` (retomavel), `tables.md`, `exemplos.md`.

Uso:
    FANNAR_CASE=letter python case_study.py --seeds 5 --pairs 100 --dice
    FANNAR_CASE=heloc python case_study.py --seeds 5 --pairs 100 --dice --compare letter satimage
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
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeClassifier, export_text

from src.attributions import (baseline_shapley, build_dice, contrastive_margin, dice_counterfactuals, dice_scores,
                              integrated_gradients, library_shap, lime_contrastive, lime_weights, margin_of,
                              minimal_flip)
from src.case import (DATASET, DESCRIPTION, IMMUTABLE, MIN_LEAF, fmt, mutable_indices, normalized, order_of, pm,
                      permutation_importance, rank_of, select_pairs, split_node)
from src.data_tabular import balanced_sample, class_names, feature_names, load_tabular, split, standardize
from src.mlp import ResidualMLP
from tabular_differences import area, flip_curve, gradxdelta
from tabular_suite import extract, train

ALL_SOURCES = ("params", "activations", "grads")
NO_OMEGA = ("params", "activations")
STEPS = (0.25, 0.5, 0.75, 1.0)
RANKINGS = ("method", "method_noomega", "shap_baseline", "shap", "shap_i", "lime", "lime_i", "ig", "dice",
            "absdiff", "gradxdelta", "random", "floor")
LABEL = {
    "method": r"método ($\Delta_f$ na Gram conjunta)",
    "method_noomega": "método sem Ω (W e A)",
    "shap_baseline": "Shapley exato com j de referência",
    "shap": "SHAP (φ(j) − φ(i), fundo de treino)",
    "shap_i": "SHAP só de i",
    "lime": "LIME contrastivo (margem j − i)",
    "lime_i": "LIME só de i",
    "ig": "gradientes integrados (x_i → x_j)",
    "dice": "DiCE (frequência de alteração)",
    "absdiff": "|Δx|",
    "gradxdelta": "grad × Δx",
    "random": "aleatória entre os que diferem",
    "floor": "método na rede não treinada",
}
CF_VARIANTS = ("seq_geo", "seq_geo_noomega", "fixed_geo", "seq_out", "random")
CF_LABEL = {"seq_geo": "sequencial, geometria", "seq_geo_noomega": "sequencial, geometria sem Ω",
            "fixed_geo": "ranking fixo, geometria", "seq_out": "sequencial, saída do modelo",
            "random": "aleatória", "dice_best": "DiCE, o mais esparso dos válidos",
            "dice_mean": "DiCE, média dos válidos", "minimal": "mínimo exato (enumeração)"}
GLOBAL = ("method", "shap", "lime", "surrogate", "label_tree")
GLOBAL_LABEL = {"method": "método (parcela média)", "shap": "SHAP global (média de |φ|)",
                "lime": "LIME global (média de |w|)", "surrogate": "árvore substituta (Gini)",
                "label_tree": "árvore nos rótulos (Gini)"}


# --------------------------------------------------------------------------
# geometria
# --------------------------------------------------------------------------


def _flat(src: dict, key: str, n: int) -> np.ndarray:
    return src[key].reshape(n, -1).astype(np.float64)


def pair_scores(model, xi, xj, device, target, shift, keys=ALL_SOURCES) -> dict:
    """Δd_f por bloco (como `tabular_differences.method_scores`), so com as fontes `keys`."""
    F = len(xi)
    swapped = np.repeat(xi[None], F, axis=0)
    swapped[np.arange(F), np.arange(F)] = xj
    rows = np.concatenate([xj[None], xi[None], swapped]).astype(np.float32)
    sources = extract(model, rows, device, target)["sources"]
    out = {}
    for b, src in sources.items():
        joint = np.ones(len(rows))
        for key in keys:
            u = _flat(src, key, len(rows))
            norms = np.linalg.norm(u, axis=1)
            joint *= ((u @ u[0]) / np.maximum(norms * norms[0], 1e-300) + shift) / (1.0 + shift)
        d = np.sqrt(np.maximum(2.0 * (1.0 - joint), 0.0))
        out[b] = d[1] - d[2:]
    return out


def set_distance(model, X: np.ndarray, S: np.ndarray, block: str, device, target, shift, keys=ALL_SOURCES) -> np.ndarray:
    """D_S(x) de cada linha de X: distancia media, na Gram conjunta do bloco, aos objetos de S."""
    rows = np.concatenate([S, X]).astype(np.float32)
    src = extract(model, rows, device, target)["sources"][block]
    joint = np.ones((len(X), len(S)))
    for key in keys:
        u = _flat(src, key, len(rows))
        u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-300)
        joint *= (u[len(S):] @ u[: len(S)].T + shift) / (1.0 + shift)
    return np.sqrt(np.maximum(2.0 * (1.0 - joint), 0.0)).mean(1)


def omega_diagnostic(model, X: np.ndarray, device, target) -> dict:
    """Por bloco e fonte: razao de participacao da Gram angular e cosseno medio fora da diagonal."""
    sources = extract(model, X.astype(np.float32), device, target)["sources"]
    out = {}
    for b, src in sources.items():
        out[b] = {}
        for key in ALL_SOURCES:
            u = _flat(src, key, len(X))
            norms = np.linalg.norm(u, axis=1)
            if np.median(norms) < 1e-12:
                out[b][key] = None  # fonte sem geometria (Ω no ultimo bloco sob alvo fixo)
                continue
            u = u / np.maximum(norms[:, None], 1e-300)
            G = u @ u.T
            lam = np.clip(np.linalg.eigvalsh(G), 0, None)
            off = G[~np.eye(len(G), dtype=bool)]
            out[b][key] = {"participation": float(lam.sum() ** 2 / max((lam ** 2).sum(), 1e-300)),
                           "mean_cos": float(off.mean()), "mean_abs_cos": float(np.abs(off).mean())}
    return out


# --------------------------------------------------------------------------
# contrafactual
# --------------------------------------------------------------------------


@torch.no_grad()
def predict(model, X: np.ndarray, device) -> np.ndarray:
    return model(torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)).argmax(1).cpu().numpy()


@torch.no_grad()
def margin(model, X: np.ndarray, device, ci: int, cj: int) -> np.ndarray:
    z = model(torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)).double().cpu().numpy()
    return z[:, cj] - z[:, ci]


def counterfactual(model, xi, mu, S, ci, cj, block, device, args, mode: str, allowed, rng, keys=ALL_SOURCES):
    """Algoritmo `alg:contrafactual` e variantes: (x', atributos alterados, valido)."""
    x = xi.astype(np.float64).copy()
    remaining = [int(f) for f in allowed if abs(mu[f] - xi[f]) > 1e-9]

    def gains(x, cand_feats):
        cand = np.repeat(x[None], len(cand_feats), axis=0)
        cand[np.arange(len(cand_feats)), cand_feats] = mu[cand_feats]
        if mode == "seq_out":
            return margin(model, cand, device, ci, cj) - margin(model, x[None], device, ci, cj)[0]
        d = set_distance(model, np.concatenate([x[None], cand]), S, block, device, args.fixed_target, args.shift, keys)
        return d[0] - d[1:]

    queue = None
    if mode == "fixed_geo":
        queue = [remaining[k] for k in order_of(gains(x, remaining), rng)]
    elif mode == "random":
        queue = [int(f) for f in rng.permutation(remaining)]
    changed = []
    while remaining:
        if queue is None:
            f = remaining[int(order_of(gains(x, remaining), rng)[0])]
        else:
            f = queue.pop(0)
        remaining.remove(f)
        changed.append(f)
        start = x[f]
        trial = np.repeat(x[None], len(STEPS), axis=0)
        trial[:, f] = start + np.asarray(STEPS) * (mu[f] - start)
        hit = np.flatnonzero(predict(model, trial, device) == cj)
        if len(hit):
            return trial[hit[0]], changed, True
        x[f] = mu[f]
    return x, changed, False


def cf_metrics(x, xi, valid, changed_n, target_tr, seconds) -> dict:
    return {"valid": bool(valid), "n_changed": int(changed_n), "l1": float(np.abs(x - xi).sum()),
            "plaus": float(np.min(np.linalg.norm(target_tr - x[None], axis=1))), "seconds": float(seconds)}


# --------------------------------------------------------------------------
# uma semente
# --------------------------------------------------------------------------


def run_one(seed: int, args, device) -> dict:
    X, y = load_tabular(DATASET)
    feats = feature_names(DATASET)
    tr, va, te = split(y, seed)
    X_tr, X_va, X_te = (a.astype(np.float32) for a in standardize(X[tr], X[va], X[te]))
    raw_tr, raw_te = X[tr], X[te]
    torch.manual_seed(seed)
    model = ResidualMLP(X.shape[1], int(y.max() + 1), args.width, args.blocks).to(device)
    untrained = copy.deepcopy(model)
    train(model, X_tr, y[tr], X_va, y[va], device, args, seed)
    model.eval()
    untrained.eval()
    preds, preds_tr = predict(model, X_te, device), predict(model, X_tr, device)
    y_te, F = y[te], X.shape[1]
    rng = np.random.default_rng(seed)
    eb = args.block
    mutable = mutable_indices(feats)

    surrogate = DecisionTreeClassifier(min_samples_leaf=args.min_leaf, random_state=seed).fit(raw_tr, preds_tr)
    label_tree = DecisionTreeClassifier(min_samples_leaf=args.min_leaf, random_state=seed).fit(raw_tr, y[tr])
    trees = {"surrogate": surrogate, "label_tree": label_tree}
    tree_preds = {k: t.predict(raw_te) for k, t in trees.items()}

    pairs = select_pairs(preds, y_te, X_te, rng, args.pairs)

    background = X_tr[rng.choice(len(X_tr), size=args.background, replace=False)]
    rows = np.unique([k for p in pairs for k in p])
    t0 = time.perf_counter()
    phi_rows = library_shap(model, X_te[rows], background, device, args.max_evals, seed)
    t_shap = (time.perf_counter() - t0) / len(rows)
    phi_at = dict(zip(rows.tolist(), phi_rows))
    lime_reg = LimeTabularExplainer(X_tr, mode="regression", discretize_continuous=False,
                                    sample_around_instance=True, random_state=seed)
    lime_cls = LimeTabularExplainer(X_tr, mode="classification", random_state=seed)
    dice = build_dice(model, X_tr, y[tr], device, seed) if args.dice else None

    grid = np.linspace(0, 1, F + 1)
    blocks = model.block_names
    curves = {r: [] for r in RANKINGS if r not in ("method", "floor") and (args.dice or r != "dice")}
    curves |= {f"method@{b}": [] for b in blocks} | {f"floor@{b}": [] for b in blocks}
    records = []
    timings = {k: [] for k in ("method", "shap_baseline", "lime", "ig", "dice")}
    for n_pair, (i, j) in enumerate(pairs):
        xi, xj = X_te[i], X_te[j]
        ci, cj = int(preds[i]), int(preds[j])
        target_tr = X_tr[preds_tr == cj].astype(np.float64)
        differ = np.flatnonzero(np.abs(xj - xi) > 1e-9)

        # ---- 1. rankings ----
        t0 = time.perf_counter()
        scores = pair_scores(model, xi, xj, device, args.fixed_target, args.shift)
        timings["method"].append(time.perf_counter() - t0)
        scores_no = pair_scores(model, xi, xj, device, args.fixed_target, args.shift, NO_OMEGA)
        scores_u = pair_scores(untrained, xi, xj, device, args.fixed_target, args.shift)
        t0 = time.perf_counter()
        phi_b, total = baseline_shapley(model, xi, xj, ci, cj, device)
        timings["shap_baseline"].append(time.perf_counter() - t0)
        assert abs(phi_b.sum() - total) < 1e-3 * max(1.0, abs(total)), "eficiencia do Shapley exato"
        t0 = time.perf_counter()
        lime_c = lime_contrastive(lime_reg, model, xi, xj, ci, cj, device, args.lime_samples)
        timings["lime"].append(time.perf_counter() - t0)
        w_i = lime_weights(lime_cls, model, xi, (ci, cj), device, args.lime_samples)
        t0 = time.perf_counter()
        ig = integrated_gradients(model, xi, xj, ci, cj, device, args.ig_steps)
        timings["ig"].append(time.perf_counter() - t0)
        attributions = {
            "shap_baseline": phi_b,
            "shap": margin_of(phi_at[j], cj, ci) - margin_of(phi_at[i], cj, ci),
            "shap_i": margin_of(phi_at[i], ci, cj),
            "lime": lime_c, "lime_i": w_i[ci] - w_i[cj], "ig": ig,
            "absdiff": np.abs(xj - xi), "gradxdelta": gradxdelta(model, xi, xj, ci, cj, device),
            "method_noomega": scores_no[eb],
        }
        if dice is not None:
            t0 = time.perf_counter()
            attributions["dice"] = dice_scores(dice, xi, cj, args.dice_cfs, rng, xj=xj)
            timings["dice"].append(time.perf_counter() - t0)
        orders = {k: np.argsort(-v, kind="stable") for k, v in attributions.items()}
        orders["random"] = np.concatenate([rng.permutation(differ), np.setdiff1d(np.arange(F), differ)])
        for b in blocks:
            orders[f"method@{b}"] = np.argsort(-scores[b], kind="stable")
            orders[f"floor@{b}"] = np.argsort(-scores_u[b], kind="stable")
        for k, order in orders.items():
            curves[k].append(flip_curve(model, xi, xj, cj, order, grid, device))

        # ---- 2. contrafactual ----
        # S: os k preditos na classe de j mais proximos de i na Gram conjunta do bloco
        cands = np.flatnonzero(preds == cj)
        rows_s = np.concatenate([X_te[[i]], X_te[cands]]).astype(np.float32)
        src = extract(model, rows_s, device, args.fixed_target)["sources"][eb]
        joint = np.ones(len(cands))
        for key in ALL_SOURCES:
            u = _flat(src, key, len(rows_s))
            u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-300)
            joint *= (u[1:] @ u[0] + args.shift) / (1.0 + args.shift)
        S = X_te[cands[np.argsort(-joint)[: args.k]]].astype(np.float64)
        mu = S.mean(0)
        cf = {"mu_is_target": bool(predict(model, mu[None], device)[0] == cj)}
        for scope, allowed in (("all", np.arange(F)),) + ((("mutable", mutable),) if IMMUTABLE else ()):
            res = {}
            for v in CF_VARIANTS:
                keys = NO_OMEGA if v == "seq_geo_noomega" else ALL_SOURCES
                mode = "seq_geo" if v == "seq_geo_noomega" else v
                t0 = time.perf_counter()
                x, changed, valid = counterfactual(model, xi, mu, S, ci, cj, eb, device, args, mode, allowed, rng, keys)
                res[v] = cf_metrics(x, xi.astype(np.float64), valid, len(changed), target_tr, time.perf_counter() - t0)
                res[v]["changed"] = changed
            t0 = time.perf_counter()
            res["minimal"] = {"n_changed": minimal_flip(model, xi, mu, cj, device, allowed),
                              "seconds": time.perf_counter() - t0}
            if dice is not None:
                t0 = time.perf_counter()
                found = dice_counterfactuals(dice, xi, cj, args.dice_cfs,
                                             None if scope == "all" else mutable.tolist())
                sec = time.perf_counter() - t0
                ok = found[predict(model, found, device) == cj] if len(found) else found
                if len(ok):
                    n = (np.abs(ok - xi[None]) > 1e-6).sum(1)
                    ms = [cf_metrics(c, xi.astype(np.float64), True, k, target_tr, sec) for c, k in zip(ok, n)]
                    res["dice_best"] = ms[int(np.argmin(n))]
                    res["dice_mean"] = {m: float(np.mean([q[m] for q in ms])) for m in ("n_changed", "l1", "plaus")}
                    res["dice_mean"] |= {"valid": True, "seconds": sec}
                else:
                    res["dice_best"] = res["dice_mean"] = {"valid": False, "seconds": sec}
            cf[scope] = res

        rec = {"i": i, "j": j, "pred_i": ci, "pred_j": cj, "n_differ": int(len(differ)), "raw_i": raw_te[i].tolist(), "raw_j": raw_te[j].tolist(),
               "method": {b: scores[b].tolist() for b in blocks}, "floor": scores_u[eb].tolist(),
               **{k: v.tolist() for k, v in attributions.items() if k != "absdiff"},
               "orders": {k: o.tolist() for k, o in orders.items()
                          if k in ("random", f"method@{eb}", f"floor@{eb}") or "@" not in k},
               "trees": {}, "cf": cf, "mu_raw": (mu * raw_tr.std(0).clip(1e-6) + raw_tr.mean(0)).tolist()}
        for name, tree in trees.items():
            node = split_node(tree, raw_te[i], raw_te[j])
            tp_i, tp_j = int(tree_preds[name][i]), int(tree_preds[name][j])
            rec["trees"][name] = {"reproduces": tp_i == ci and tp_j == cj, "node": node,
                                  "feature": None if node is None else int(tree.tree_.feature[node]),
                                  "threshold": None if node is None else float(tree.tree_.threshold[node])}
        if len(differ) >= 3:
            rec["spearman_method_shap"] = float(spearmanr(scores[eb][differ], phi_b[differ])[0])
        records.append(rec)
        if (n_pair + 1) % 20 == 0:
            print(f"  semente {seed}: {n_pair + 1}/{len(pairs)} pares", flush=True)

    mean_curves = {k: np.mean(v, 0).tolist() for k, v in curves.items()}
    sample = balanced_sample(y_te, total=args.global_samples, seed=seed)
    phi_g = library_shap(model, X_te[sample], background, device, args.max_evals, seed)
    shap_global = np.abs(contrastive_margin(phi_g, preds[sample])).mean(0)
    lime_global = np.zeros(F)
    for k in sample:
        lime_global += np.abs(lime_weights(lime_cls, model, X_te[k], (int(preds[k]),), device,
                                           args.lime_samples)[int(preds[k])])
    lime_global /= len(sample)
    method_global = {b: np.mean([normalized(r["method"][b]) for r in records], 0).tolist() for b in blocks}
    perm = permutation_importance(model, X_te, preds, device, rng, args.repeats)
    diag = {"trained": omega_diagnostic(model, X_te[sample], device, args.fixed_target),
            "untrained": omega_diagnostic(untrained, X_te[sample], device, args.fixed_target)}

    return {
        "seed": seed, "blocks": blocks, "grid": grid.tolist(), "n_features": F, "k": args.k, "min_leaf": args.min_leaf,
        "accuracy": float((preds == y_te).mean()),
        "fidelity": {k: float((tree_preds[k] == preds).mean()) for k in trees},
        "tree_accuracy": {k: float((tree_preds[k] == y_te).mean()) for k in trees},
        "curves": mean_curves, "auc": {k: area(np.asarray(v), grid) for k, v in mean_curves.items()},
        "pairs": records, "omega": diag,
        "global": {"method": method_global, "shap": shap_global.tolist(), "lime": lime_global.tolist(),
                   "surrogate": surrogate.feature_importances_.tolist(),
                   "label_tree": label_tree.feature_importances_.tolist(), "permutation": perm.tolist()},
        "cost": {"method_s": float(np.mean(timings["method"])),
                 "shap_baseline_s": float(np.mean(timings["shap_baseline"])), "shap_s_per_sample": float(t_shap),
                 "lime_s": float(np.mean(timings["lime"])), "ig_s": float(np.mean(timings["ig"])),
                 "dice_s": float(np.mean(timings["dice"])) if timings["dice"] else None,
                 "method_evals": F + 2, "shap_baseline_evals": 2 ** F},
        "surrogate_top": export_text(surrogate, feature_names=feats, max_depth=3),
    }


def refit_trees(run: dict, args, device) -> dict:
    """Reajusta as arvores de corroboracao de uma execucao gravada com `--min-leaf`, sem refazer os pares.

    O MLP e retreinado com a mesma semente (a mesma receita de `run_one`), para que a arvore substituta
    seja ajustada as mesmas predicoes; os nos de separacao sao recalculados sobre os pares gravados.
    """
    seed = run["seed"]
    X, y = load_tabular(DATASET)
    tr, va, te = split(y, seed)
    X_tr, X_va, X_te = (a.astype(np.float32) for a in standardize(X[tr], X[va], X[te]))
    raw_tr, raw_te = X[tr], X[te]
    torch.manual_seed(seed)
    model = ResidualMLP(X.shape[1], int(y.max() + 1), args.width, args.blocks).to(device)
    train(model, X_tr, y[tr], X_va, y[va], device, args, seed)
    model.eval()
    preds, preds_tr = predict(model, X_te, device), predict(model, X_tr, device)
    assert abs(float((preds == y[te]).mean()) - run["accuracy"]) < 1e-9, "o MLP retreinado difere do gravado"
    trees = {"surrogate": DecisionTreeClassifier(min_samples_leaf=args.min_leaf, random_state=seed).fit(raw_tr, preds_tr),
             "label_tree": DecisionTreeClassifier(min_samples_leaf=args.min_leaf, random_state=seed).fit(raw_tr, y[tr])}
    tree_preds = {k: t.predict(raw_te) for k, t in trees.items()}
    for rec in run["pairs"]:
        i, j = rec["i"], rec["j"]
        ci, cj = int(preds[i]), int(preds[j])
        for name, tree in trees.items():
            node = split_node(tree, raw_te[i], raw_te[j])
            rec["trees"][name] = {"reproduces": int(tree_preds[name][i]) == ci and int(tree_preds[name][j]) == cj,
                                  "node": node,
                                  "feature": None if node is None else int(tree.tree_.feature[node]),
                                  "threshold": None if node is None else float(tree.tree_.threshold[node])}
    run["fidelity"] = {k: float((tree_preds[k] == preds).mean()) for k in trees}
    run["tree_accuracy"] = {k: float((tree_preds[k] == y[te]).mean()) for k in trees}
    run["global"]["surrogate"] = trees["surrogate"].feature_importances_.tolist()
    run["global"]["label_tree"] = trees["label_tree"].feature_importances_.tolist()
    run["surrogate_top"] = export_text(trees["surrogate"], feature_names=feature_names(DATASET), max_depth=3)
    run["min_leaf"] = args.min_leaf
    return run


def compare_diagnostic(name: str, args, device) -> dict:
    """O diagnostico de Ω numa base multiclasse da suite (semente 0), para contraste."""
    X, y = load_tabular(name)
    tr, va, te = split(y, 0)
    X_tr, X_va, X_te = (a.astype(np.float32) for a in standardize(X[tr], X[va], X[te]))
    torch.manual_seed(0)
    model = ResidualMLP(X.shape[1], int(y.max() + 1), args.width, args.blocks).to(device)
    train(model, X_tr, y[tr], X_va, y[va], device, args, 0)
    model.eval()
    sample = balanced_sample(y[te], total=args.global_samples, seed=0)
    return omega_diagnostic(model, X_te[sample], device, args.fixed_target)


# --------------------------------------------------------------------------
# relatorio
# --------------------------------------------------------------------------


def local_tree_metrics(runs: list[dict], tree: str, block: str) -> dict:
    """Posicao do atributo em que a arvore separa i de j, em cada ranking, por semente."""
    keys = {"method": f"method@{block}", "floor": f"floor@{block}",
            **{k: k for k in RANKINGS if k not in ("method", "floor")}}
    keys = {k: v for k, v in keys.items() if v in runs[0]["pairs"][0]["orders"]}
    out = {k: {"top1": [], "top3": []} for k in keys} | {"chance": {"top1": [], "top3": []}}
    counts = []
    for r in runs:
        per = {k: {"top1": [], "top3": []} for k in out}
        for p in r["pairs"]:
            t = p["trees"][tree]
            if not t["reproduces"] or t["feature"] is None:
                continue
            f, n = t["feature"], p["n_differ"]
            for k, key in keys.items():
                pos = rank_of(np.asarray(p["orders"][key]), f)
                per[k]["top1"].append(pos == 0)
                per[k]["top3"].append(pos < 3)
            per["chance"]["top1"].append(1.0 / n)
            per["chance"]["top3"].append(min(3, n) / n)
        counts.append(len(per["chance"]["top1"]))
        for k in out:
            for m in out[k]:
                out[k][m].append(np.mean(per[k][m]) if per[k][m] else np.nan)
    return {"metrics": out, "used": counts}


def write_report(runs: list[dict], args, out: Path, compare: dict) -> None:
    feats, classes = feature_names(DATASET), class_names(DATASET)
    blocks, eb = runs[0]["blocks"], args.block
    P = [p for r in runs for p in r["pairs"]]
    L = [f"# Estudo de caso de P6: `{DATASET}`\n",
         f"{len(runs)} sementes × {args.pairs} pares. MLP residual de {args.blocks} blocos; método na Gram conjunta "
         f"de {eb}, alvo fixo (classe {classes[args.fixed_target]}). Média ± desvio entre sementes.\n",
         f"- MLP: acerto no teste {pm([r['accuracy'] for r in runs])}",
         f"- árvore substituta: fidelidade {pm([r['fidelity']['surrogate'] for r in runs])}; árvore nos rótulos: "
         f"acerto {pm([r['tree_accuracy']['label_tree'] for r in runs])}",
         f"- atributos que diferem entre i e j: {pm([p['n_differ'] for p in P], 1)} de {runs[0]['n_features']}\n"]

    keys = [f"method@{eb}", "method_noomega", "shap_baseline", "shap", "shap_i", "lime", "lime_i", "ig", "dice",
            "absdiff", "gradxdelta", "random", f"floor@{eb}"]
    keys = [k for k in keys if k in runs[0]["auc"]]
    L += ["## 1. Teste contrafactual de trocas (área sob a curva)\n", "| ranking | AUC | método vence em |", "|---|---|---|"]
    for k in keys:
        base = k.split("@")[0]
        wins = "—" if base == "method" else f"{np.mean([r['auc'][f'method@{eb}'] > r['auc'][k] for r in runs]):.0%}"
        L.append(f"| {LABEL[base]} | {pm([r['auc'][k] for r in runs])} | {wins} |")
    L += ["", "Método por bloco:\n", "| | " + " | ".join(blocks) + " |", "|---|" + "---|" * len(blocks)]
    for kind in ("method", "floor"):
        L.append(f"| {LABEL[kind]} | " + " | ".join(pm([r['auc'][f'{kind}@{b}'] for r in runs]) for b in blocks) + " |")
    sp = [p["spearman_method_shap"] for p in P if "spearman_method_shap" in p and np.isfinite(p["spearman_method_shap"])]
    top1 = [np.argmax(p["method"][eb]) == np.argmax(p["shap_baseline"]) for p in P]
    L += ["", f"Método × Shapley exato, par a par: Spearman mediano {fmt(np.median(sp))}; mesmo 1º atributo em "
          f"{np.mean(top1):.0%} dos pares.\n"]
    c = {k: np.mean([r["cost"][k] for r in runs]) for k in ("method_s", "shap_baseline_s", "shap_s_per_sample", "lime_s", "ig_s")}
    L += ["Custo por par (GPU): " + ", ".join(f"{k} {fmt(v * 1e3, 1)} ms" for k, v in c.items())
          + (f", dice_s {fmt(np.mean([r['cost']['dice_s'] for r in runs]) * 1e3, 1)} ms" if runs[0]["cost"]["dice_s"] else "") + "\n"]

    for tree, title in (("surrogate", "árvore substituta"), ("label_tree", "árvore nos rótulos")):
        res = local_tree_metrics(runs, tree, eb)
        L += [f"## 2. Atributo em que a {title} separa i de j\n",
              f"Pares usados por semente: {pm(res['used'], 1)}.\n", "| ranking | 1º | entre os 3 primeiros |", "|---|---|---|"]
        for k, m in res["metrics"].items():
            L.append(f"| {'acaso' if k == 'chance' else LABEL[k]} | {pm(m['top1'])} | {pm(m['top3'])} |")
        L.append("")

    L += ["## 3. Construção do contrafactual\n",
          f"Alvo: média dos {args.k} objetos preditos na classe de j mais próximos de i na Gram conjunta de {eb}. "
          f"A média já é da classe de j em {pm([np.mean([p['cf'].get('mu_is_target', p['cf'].get('mu_is_good')) for p in r['pairs']]) for r in runs])} dos pares. "
          f"Imutáveis no escopo 'mutable': {', '.join(IMMUTABLE)}.\n"]
    for scope in [s for s in ("all", "mutable") if s in runs[0]["pairs"][0]["cf"]]:
        L += [f"### Escopo: {'todos os atributos' if scope == 'all' else 'só os mutáveis'}\n",
              "| procedimento | válido | alterados | atinge o mínimo | L1 | plausibilidade | ms |", "|---|---|---|---|---|---|---|"]
        for v in (*CF_VARIANTS, "dice_best", "dice_mean"):
            if v not in runs[0]["pairs"][0]["cf"][scope]:
                continue

            def stat(key, prs, v=v):
                vals = [p["cf"][scope][v].get(key) for p in prs if p["cf"][scope][v]["valid"]]
                return np.mean(vals) if vals else np.nan

            def hits(prs, v=v):
                ok = [p for p in prs if p["cf"][scope][v]["valid"] and p["cf"][scope]["minimal"]["n_changed"] is not None]
                return np.mean([p["cf"][scope][v]["n_changed"] == p["cf"][scope]["minimal"]["n_changed"] for p in ok]) if ok else np.nan
            L.append(f"| {CF_LABEL[v]} | {pm([np.mean([p['cf'][scope][v]['valid'] for p in r['pairs']]) for r in runs])} | "
                     f"{pm([stat('n_changed', r['pairs']) for r in runs], 2)} | {pm([hits(r['pairs']) for r in runs])} | "
                     f"{pm([stat('l1', r['pairs']) for r in runs], 2)} | {pm([stat('plaus', r['pairs']) for r in runs], 2)} | "
                     f"{pm([stat('seconds', r['pairs']) * 1e3 for r in runs], 1)} |")
        mins = [[p["cf"][scope]["minimal"]["n_changed"] for p in r["pairs"]] for r in runs]
        L.append(f"| {CF_LABEL['minimal']} | {pm([np.mean([m is not None for m in ms]) for ms in mins])} | "
                 f"{pm([np.mean([m for m in ms if m is not None]) for ms in mins], 2)} | 100% | — | — | — |\n")

    L += ["## 4. Diagnóstico da fonte Ω (amostra balanceada do teste)\n",
          "Razão de participação da Gram angular (dimensão efetiva) e cosseno médio entre objetos distintos.\n",
          "| base | modelo | fonte | " + " | ".join(blocks) + " |", "|---|---|---|" + "---|" * len(blocks)]

    def diag_rows(label, diags, model):
        for key in ALL_SOURCES:
            for metric in ("participation", "mean_cos"):
                vals = []
                for b in blocks:
                    v = [d[b][key][metric] for d in diags if d[b][key] is not None]
                    vals.append(fmt(np.mean(v), 2) if v else "—")
                L.append(f"| {label} | {model} | {key} ({metric}) | " + " | ".join(vals) + " |")
    diag_rows(DATASET, [r["omega"]["trained"] for r in runs], "treinado")
    diag_rows(DATASET, [r["omega"]["untrained"] for r in runs], "não treinado")
    for name, d in compare.items():
        diag_rows(name, [d], "treinado")
    L.append("")

    perm = np.array([r["global"]["permutation"] for r in runs])
    L += ["## 5. Importância global contra a permutação no MLP\n", "| fonte | Spearman |", "|---|---|"]
    for k in GLOBAL:
        vals = [np.asarray(r["global"][k][eb] if k == "method" else r["global"][k]) for r in runs]
        L.append(f"| {GLOBAL_LABEL[k]} | {pm([spearmanr(v, p)[0] for v, p in zip(vals, perm)])} |")
    order = np.argsort(-perm.mean(0))
    pn = np.array([normalized(p) for p in perm]).mean(0)
    L += ["", "| atributo | significado | imutável | permutação |", "|---|---|---|---|"]
    for f in order:
        L.append(f"| {feats[f]} | {DESCRIPTION[feats[f]]} | {'sim' if feats[f] in IMMUTABLE else ''} | {fmt(pn[f])} |")
    L += ["", "## Topo da árvore substituta (semente 0)\n", "```", runs[0]["surrogate_top"], "```"]
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    write_examples(runs[0], args, out, feats)


def write_examples(run: dict, args, out: Path, feats) -> None:
    """Contrafactuais da semente 0 lidos por extenso, na escala original."""
    E = [f"# Contrafactuais do método — `{DATASET}`, semente 0\n",
         "Objeto i e o contrafactual construído pelo Algoritmo `alg:contrafactual` só com os atributos "
         "mutáveis; valores na escala original.\n"]
    shown = 0
    for p in run["pairs"]:
        scope = "mutable" if "mutable" in p["cf"] else "all"
        res = p["cf"][scope]["seq_geo"]
        if shown >= args.examples or not res["valid"]:
            continue
        shown += 1
        E += [f"## Par {p['i']} → {p['j']}: {res['n_changed']} atributo(s) alterado(s) "
              f"(mínimo {p['cf'][scope]['minimal']['n_changed']})\n",
              "| atributo | significado | i | alvo (média dos vizinhos da classe de j) |", "|---|---|---|---|"]
        for f in res["changed"]:
            E.append(f"| {feats[f]} | {DESCRIPTION[feats[f]]} | {p['raw_i'][f]:.0f} | {p['mu_raw'][f]:.1f} |")
        E.append("")
    (out / "exemplos.md").write_text("\n".join(E) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--block", default="b3")
    ap.add_argument("--k", type=int, default=5, help="vizinhos Good cuja media e o alvo do contrafactual")
    ap.add_argument("--background", type=int, default=100)
    ap.add_argument("--max-evals", type=int, default=1000)
    ap.add_argument("--global-samples", type=int, default=500)
    ap.add_argument("--lime-samples", type=int, default=5000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--ig-steps", type=int, default=64)
    ap.add_argument("--dice", action="store_true", help="inclui o DiCE (pip install dice-ml)")
    ap.add_argument("--dice-cfs", type=int, default=5)
    ap.add_argument("--compare", nargs="*", default=[], help="bases multiclasse para o contraste do diagnóstico de Ω")
    ap.add_argument("--min-leaf", type=int, default=MIN_LEAF, help="o de `src.case.MIN_LEAF`, escolhido pela validacao")
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--fixed-target", type=int, default=0)
    ap.add_argument("--only", type=int, nargs="*", default=None,
                    help="roda so estas sementes, sem relatorio (para paralelizar em varios processos)")
    ap.add_argument("--refit-trees", action="store_true",
                    help="reajusta as arvores de corroboracao das execucoes gravadas com outro --min-leaf")
    ap.add_argument("--out", default=f"outputs/{DATASET}_case")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = []
    for seed in (range(args.seeds) if args.only is None else args.only):
        path = out / "runs" / f"seed{seed}.json"
        if path.exists():
            run = json.loads(path.read_text(encoding="utf-8"))
            if args.refit_trees and run.get("min_leaf", 50) != args.min_leaf:  # sem o campo: a rodada de folha 50
                run = refit_trees(run, args, device)
                path.write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
                print(f"semente {seed}: árvores reajustadas (folha mínima {args.min_leaf})", flush=True)
            runs.append(run)
            continue
        t0 = time.perf_counter()
        r = run_one(seed, args, device)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        runs.append(r)
        a = r["auc"]
        print(f"semente {seed} ({time.perf_counter() - t0:.0f} s): acerto {r['accuracy']:.3f} | AUC método "
              f"{a[f'method@{args.block}']:.3f} sem Ω {a['method_noomega']:.3f} Shapley {a['shap_baseline']:.3f} "
              f"IG {a['ig']:.3f} |Δx| {a['absdiff']:.3f}", flush=True)
    if args.only is not None:
        return
    cpath = out / "compare.json"
    compare = json.loads(cpath.read_text(encoding="utf-8")) if cpath.exists() else {}
    for name in args.compare:
        if name not in compare:
            compare[name] = compare_diagnostic(name, args, device)
            cpath.write_text(json.dumps(compare), encoding="utf-8")
    write_report(runs, args, out, compare)
    print(f"pronto: {out}/")


if __name__ == "__main__":
    main()

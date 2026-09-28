"""Estudo de caso na `letter`: o que o metodo diz sobre o MLP, comparado ao SHAP e a arvores.

Por que a `letter`. Das dez bases da suite, e a que junta as tres condicoes do estudo: os 16
atributos tem significado (Frey & Slate, 1991: caixa da letra, pixels acesos, momentos e
contagens de bordas), tem 26 classes (a margem contrastiva nao e tautologica como no caso
binario) e o treinamento reorganiza a geometria (CKA treinada × nao treinada 0,54; no
Titanic, 0,91, e la o piso alcanca o metodo). Com F = 16, o Shapley exato de um par custa
2^16 avaliacoes e cabe na GPU -- a comparacao com o SHAP nao depende de amostragem.

Tres perguntas.

1. **Metodo × SHAP no mesmo par.** Para (i, j) com predicoes diferentes (mesma escolha de
   pares de `tabular_differences.py`), cada ranking de atributos passa pelo teste
   contrafactual de trocas. Variantes do SHAP, da mais forte para a mais usada:

     shap_baseline  Shapley **exato** do jogo v(S) = m(x_j em S, x_i fora) − m(x_i), com
                    m = f_{c_j} − f_{c_i} (baseline Shapley, Sundararajan & Najmi 2020). E a
                    decomposicao exata da propria troca que o teste mede -- o "teto" do SHAP.
     shap           SHAP intervencional da biblioteca (PermutationExplainer, fundo de 100
                    amostras de treino): φ_m(x_j) − φ_m(x_i), que tambem soma m(x_j) − m(x_i).
     shap_i         SHAP so da amostra i, φ_{c_i − c_j}(x_i): o uso mais comum ("por que i e c_i").
     lime           LIME (Ribeiro et al., 2016) em regressao sobre a margem f_{c_j} − f_{c_i}, amostrando
                    em torno de x_i: w_f (x_{j,f} − x_{i,f})/σ_f, a mudanca que o modelo linear local
                    preve ao levar o atributo ao valor de j.
     lime_i         LIME de classificacao usual so de i (atributos discretizados): w_{c_i} − w_{c_j}.

   Referencias: |Δx|, grad×Δx, ordem aleatoria **entre os atributos que diferem** (na `letter`
   os atributos sao inteiros de 0 a 15 e o vizinho mais proximo repete varios valores) e o
   metodo na rede nao treinada.

2. **Metodo × arvore, par a par.** Uma arvore **substituta** e ajustada as predicoes do MLP
   no treino (fidelidade medida no teste). Nos pares em que ela reproduz as duas predicoes
   do MLP, o no em que os caminhos de i e j se separam da o atributo que, para a arvore,
   distingue os dois. Mede-se em que posicao de cada ranking esse atributo aparece. A
   mesma conta e feita com uma arvore treinada nos rotulos (modelo independente do MLP).

3. **Importancia global.** Parcela media de cada atributo no metodo (sobre os pares), SHAP
   global (media de |φ| da margem da classe predita), importancia das duas arvores, todas
   comparadas por Spearman com a importancia por permutacao do MLP (queda na concordancia
   com a propria predicao), que e a referencia externa do que o MLP usa.

Custos: o metodo precisa de F + 2 linhas numa passagem direta e reversa; o Shapley exato,
de 2^F passagens diretas; o SHAP da biblioteca, de max_evals × |fundo| por amostra.

Saidas em `outputs/letter_case/`: `runs/seed{s}.json` (retomavel), `tables.md`,
`exemplos.md`, `case_study.pdf/png`.

Uso:
    python letter_case_study.py --seeds 5 --pairs 100
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from math import factorial
from pathlib import Path

import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
import torch
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeClassifier, export_text

from src.data_tabular import balanced_sample, class_names, feature_names, load_tabular, split, standardize
from src.mlp import ResidualMLP
from tabular_differences import area, flip_curve, gradxdelta, method_scores
from tabular_suite import train

DATASET = "letter"

# Frey & Slate (1991), "Letter recognition using Holland-style adaptive classifiers".
DESCRIPTION = {
    "x-box": "posição horizontal da caixa",
    "y-box": "posição vertical da caixa",
    "width": "largura da caixa",
    "high": "altura da caixa",
    "onpix": "nº de pixels acesos",
    "x-bar": "x médio dos pixels acesos",
    "y-bar": "y médio dos pixels acesos",
    "x2bar": "variância de x",
    "y2bar": "variância de y",
    "xybar": "correlação entre x e y",
    "x2ybr": "média de x²y",
    "xy2br": "média de xy²",
    "x-ege": "nº médio de bordas, varrendo da esquerda para a direita",
    "xegvy": "correlação de x-ege com y",
    "y-ege": "nº médio de bordas, varrendo de baixo para cima",
    "yegvx": "correlação de y-ege com x",
}

RANKINGS = ("method", "shap_baseline", "shap", "shap_i", "lime", "lime_i", "absdiff", "gradxdelta", "random", "floor")
LABEL = {
    "method": "método (Δd na Gram conjunta)",
    "shap_baseline": "Shapley exato com j de referência",
    "shap": "SHAP (φ(j) − φ(i), fundo de treino)",
    "shap_i": "SHAP só de i",
    "lime": "LIME contrastivo (margem j − i)",
    "lime_i": "LIME só de i",
    "absdiff": "|Δx|",
    "gradxdelta": "grad × Δx",
    "random": "aleatória entre os que diferem",
    "floor": "método na rede não treinada",
}
GLOBAL = ("method", "shap", "lime", "surrogate", "label_tree")
GLOBAL_LABEL = {"method": "método (parcela média)", "shap": "SHAP global (média de |φ|)",
                "lime": "LIME global (média de |w|)",
                "surrogate": "árvore substituta (Gini)", "label_tree": "árvore nos rótulos (Gini)",
                "permutation": "permutação no MLP (referência)"}


# --------------------------------------------------------------------------
# atribuicoes
# --------------------------------------------------------------------------


def baseline_shapley(model, x_from, x_to, c_from: int, c_to: int, device, batch: int = 1 << 16):
    """Shapley exato de v(S) = m(x_to em S, x_from fora) − m(x_from), m = f_{c_to} − f_{c_from}.

    Devolve (φ, v(todos) − v(vazio)); a soma de φ e exatamente o segundo valor (eficiencia).
    """
    F = len(x_from)
    masks = torch.arange(1 << F, device=device)
    bits = ((masks[:, None] >> torch.arange(F, device=device)) & 1).bool()
    a = torch.as_tensor(x_from, dtype=torch.float32, device=device)
    b = torch.as_tensor(x_to, dtype=torch.float32, device=device)
    values = []
    with torch.no_grad():
        for s in range(0, len(bits), batch):
            logits = model(torch.where(bits[s : s + batch], b, a))
            values.append((logits[:, c_to] - logits[:, c_from]).double())
    v = torch.cat(values)
    size = bits.sum(1)
    w = torch.tensor([factorial(k) * factorial(F - k - 1) / factorial(F) for k in range(F)],
                     dtype=torch.float64, device=device)
    phi = np.empty(F)
    for f in range(F):
        without = masks[~bits[:, f]]
        phi[f] = float((w[size[without]] * (v[without | (1 << f)] - v[without])).sum())
    return phi, float(v[-1] - v[0])


def library_shap(model, X: np.ndarray, background: np.ndarray, device, max_evals: int, seed: int) -> np.ndarray:
    """SHAP intervencional da biblioteca para todos os logits: (n, F, K)."""
    def f(x):
        with torch.no_grad():
            return model(torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)).cpu().numpy()
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.PermutationExplainer(f, masker, seed=seed)
    return explainer(X, max_evals=max_evals, silent=True).values


def logits_fn(model, device):
    def f(x):
        with torch.no_grad():
            return model(torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)).double().cpu().numpy()
    return f


def lime_contrastive(explainer: LimeTabularExplainer, model, xi, xj, ci: int, cj: int, device,
                     num_samples: int) -> np.ndarray:
    """LIME em regressao sobre a margem m = f_{c_j} − f_{c_i}, amostrando em torno de x_i.

    O modelo linear local da m ~ Σ_f w_f (x_f − x_{i,f})/σ_f; a previsao da mudanca de m ao levar
    cada atributo ao valor de j e w_f (x_{j,f} − x_{i,f})/σ_f, que e a pontuacao do atributo.
    """
    f = logits_fn(model, device)
    exp = explainer.explain_instance(xi, lambda X: f(X)[:, cj] - f(X)[:, ci], num_features=len(xi),
                                     num_samples=num_samples)
    w = np.zeros(len(xi))
    # no modo de regressao o LIME guarda em local_exp[1] os pesos e em local_exp[0] os mesmos
    # pesos com o sinal trocado (lime_tabular.py); a chave 1 e a do modelo linear ajustado
    for feat, weight in exp.local_exp[1]:
        w[feat] = weight
    return w * (xj - xi) / explainer.scaler.scale_


def lime_weights(explainer: LimeTabularExplainer, model, x, labels, device, num_samples: int) -> dict:
    """LIME de classificacao usual (atributos discretizados): pesos por classe pedida."""
    f = logits_fn(model, device)

    def proba(X):
        z = f(X)
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)
    exp = explainer.explain_instance(x, proba, labels=tuple(labels), num_features=len(x),
                                     num_samples=num_samples)
    out = {}
    for c in labels:
        w = np.zeros(len(x))
        for feat, weight in exp.local_exp[c]:
            w[feat] = weight
        out[c] = w
    return out


def margin_of(phi: np.ndarray, c_to: int, c_from: int) -> np.ndarray:
    """Atribuicao da diferenca de logits f_{c_to} − f_{c_from}, por linearidade do Shapley."""
    return phi[..., c_to] - phi[..., c_from]


def contrastive_margin(phi: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Atribuicao da margem um-contra-o-resto da classe c de cada amostra: (n, F)."""
    K = phi.shape[-1]
    own = phi[np.arange(len(c)), :, c]
    return own - (phi.sum(-1) - own) / (K - 1)


# --------------------------------------------------------------------------
# arvores
# --------------------------------------------------------------------------


def split_node(tree: DecisionTreeClassifier, a: np.ndarray, b: np.ndarray):
    """No em que os caminhos de a e b se separam (None se caem na mesma folha).

    O sklearn numera os nos em pre-ordem, entao todo ancestral tem indice menor: o ancestral
    comum mais profundo e o maior indice comum aos dois caminhos.
    """
    path = tree.decision_path(np.stack([a, b]).astype(np.float32))
    pa, pb = path[0].indices, path[1].indices
    common = np.intersect1d(pa, pb)
    node = int(common.max())
    if node == pa[-1] and node == pb[-1]:
        return None
    return node


def rank_of(order: np.ndarray, feature: int) -> int:
    return int(np.flatnonzero(order == feature)[0])


def permutation_importance(model, X: np.ndarray, preds: np.ndarray, device, rng, repeats: int) -> np.ndarray:
    """Queda na concordancia do MLP com a propria predicao ao permutar cada atributo."""
    out = np.zeros(X.shape[1])
    with torch.no_grad():
        for f in range(X.shape[1]):
            for _ in range(repeats):
                Xp = X.copy()
                Xp[:, f] = rng.permutation(Xp[:, f])
                p = model(torch.from_numpy(Xp).to(device)).argmax(1).cpu().numpy()
                out[f] += 1.0 - (p == preds).mean()
    return out / repeats


def normalized(v: np.ndarray) -> np.ndarray:
    v = np.maximum(np.asarray(v, dtype=float), 0)
    return v / max(v.sum(), 1e-12)


# --------------------------------------------------------------------------
# uma semente
# --------------------------------------------------------------------------


def run_one(seed: int, args, device) -> dict:
    X, y = load_tabular(DATASET)
    tr, va, te = split(y, seed)
    X_tr, X_va, X_te = (a.astype(np.float32) for a in standardize(X[tr], X[va], X[te]))
    raw_tr, raw_te = X[tr], X[te]
    torch.manual_seed(seed)
    model = ResidualMLP(X.shape[1], int(y.max() + 1), args.width, args.blocks).to(device)
    untrained = copy.deepcopy(model)
    train(model, X_tr, y[tr], X_va, y[va], device, args, seed)
    model.eval()
    untrained.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_te).to(device)).argmax(1).cpu().numpy()
        preds_tr = model(torch.from_numpy(X_tr).to(device)).argmax(1).cpu().numpy()
    y_te = y[te]
    F = X.shape[1]
    rng = np.random.default_rng(seed)

    # arvores: nos atributos brutos, para os limiares ficarem na escala original (0 a 15)
    surrogate = DecisionTreeClassifier(min_samples_leaf=args.min_leaf, random_state=seed).fit(raw_tr, preds_tr)
    label_tree = DecisionTreeClassifier(min_samples_leaf=args.min_leaf, random_state=seed).fit(raw_tr, y[tr])
    trees = {"surrogate": surrogate, "label_tree": label_tree}
    tree_preds = {k: t.predict(raw_te) for k, t in trees.items()}

    # pares: mesma regra de `tabular_differences.py`
    pairs = []
    for i in rng.choice(np.flatnonzero(preds == y_te), size=args.pairs, replace=False):
        other = np.flatnonzero(preds != preds[i])
        j = int(other[np.argmin(np.linalg.norm(X_te[other] - X_te[i], axis=1))])
        pairs.append((int(i), j))

    # SHAP da biblioteca: uma chamada para todas as amostras dos pares
    background = X_tr[rng.choice(len(X_tr), size=args.background, replace=False)]
    rows = np.unique([k for p in pairs for k in p])
    t0 = time.perf_counter()
    phi_rows = library_shap(model, X_te[rows], background, device, args.max_evals, seed)
    t_shap = (time.perf_counter() - t0) / len(rows)
    phi_at = dict(zip(rows.tolist(), phi_rows))
    lime_reg = LimeTabularExplainer(X_tr, mode="regression", discretize_continuous=False,
                                    sample_around_instance=True, random_state=seed)
    lime_cls = LimeTabularExplainer(X_tr, mode="classification", random_state=seed)

    grid = np.linspace(0, 1, args.steps + 1)
    blocks = model.block_names
    curves = {r: [] for r in RANKINGS if r not in ("method", "floor")}
    curves |= {f"method@{b}": [] for b in blocks} | {f"floor@{b}": [] for b in blocks}
    records, timings = [], {"method": [], "shap_baseline": [], "lime": []}
    for i, j in pairs:
        xi, xj, ci, cj = X_te[i], X_te[j], int(preds[i]), int(preds[j])
        differ = np.flatnonzero(np.abs(xj - xi) > 1e-9)

        t0 = time.perf_counter()
        scores = method_scores(model, xi, xj, device, args.fixed_target, args.shift)
        timings["method"].append(time.perf_counter() - t0)
        scores_u = method_scores(untrained, xi, xj, device, args.fixed_target, args.shift)
        t0 = time.perf_counter()
        phi_b, total = baseline_shapley(model, xi, xj, ci, cj, device)
        timings["shap_baseline"].append(time.perf_counter() - t0)
        assert abs(phi_b.sum() - total) < 1e-3 * max(1.0, abs(total)), "eficiencia do Shapley exato"

        t0 = time.perf_counter()
        lime_c = lime_contrastive(lime_reg, model, xi, xj, ci, cj, device, args.lime_samples)
        timings["lime"].append(time.perf_counter() - t0)
        w_i = lime_weights(lime_cls, model, xi, (ci, cj), device, args.lime_samples)

        shuffled = rng.permutation(differ)
        rest = np.setdiff1d(np.arange(F), differ)
        attributions = {
            "shap_baseline": phi_b,
            "shap": margin_of(phi_at[j], cj, ci) - margin_of(phi_at[i], cj, ci),
            "shap_i": margin_of(phi_at[i], ci, cj),
            "lime": lime_c,
            "lime_i": w_i[ci] - w_i[cj],
            "absdiff": np.abs(xj - xi),
            "gradxdelta": gradxdelta(model, xi, xj, ci, cj, device),
        }
        orders = {k: np.argsort(-v, kind="stable") for k, v in attributions.items()}
        orders["random"] = np.concatenate([shuffled, rest])
        for b in blocks:
            orders[f"method@{b}"] = np.argsort(-scores[b], kind="stable")
            orders[f"floor@{b}"] = np.argsort(-scores_u[b], kind="stable")
        for k, order in orders.items():
            curves[k].append(flip_curve(model, xi, xj, cj, order, grid, device))

        eb = args.block
        rec = {"i": i, "j": j, "pred_i": ci, "pred_j": cj, "true_i": int(y_te[i]), "true_j": int(y_te[j]),
               "n_differ": int(len(differ)), "raw_i": raw_te[i].tolist(), "raw_j": raw_te[j].tolist(),
               "method": {b: scores[b].tolist() for b in blocks}, "floor": scores_u[eb].tolist(),
               **{k: v.tolist() for k, v in attributions.items() if k != "absdiff"},
               "orders": {k: o.tolist() for k, o in orders.items()
                          if k in ("random", f"method@{eb}", f"floor@{eb}") or "@" not in k},
               "trees": {}}
        for name, tree in trees.items():
            node = split_node(tree, raw_te[i], raw_te[j])
            tp_i, tp_j = int(tree_preds[name][i]), int(tree_preds[name][j])
            rec["trees"][name] = {
                "pred_i": tp_i, "pred_j": tp_j, "reproduces": tp_i == ci and tp_j == cj,
                "node": node,
                "feature": None if node is None else int(tree.tree_.feature[node]),
                "threshold": None if node is None else float(tree.tree_.threshold[node]),
            }
        # concordancia metodo × Shapley exato, so entre os atributos que diferem
        if len(differ) >= 3:
            rec["spearman_method_shap"] = float(spearmanr(scores[eb][differ], phi_b[differ])[0])
        records.append(rec)

    mean_curves = {k: np.mean(v, 0).tolist() for k, v in curves.items()}

    # importancia global
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

    return {
        "seed": seed, "blocks": blocks, "grid": grid.tolist(), "n_features": F,
        "accuracy": float((preds == y_te).mean()),
        "fidelity": {k: float((tree_preds[k] == preds).mean()) for k in trees},
        "tree_accuracy": {k: float((tree_preds[k] == y_te).mean()) for k in trees},
        "tree_leaves": {k: int(t.get_n_leaves()) for k, t in trees.items()},
        "tree_depth": {k: int(t.get_depth()) for k, t in trees.items()},
        "curves": mean_curves, "auc": {k: area(np.asarray(v), grid) for k, v in mean_curves.items()},
        "pairs": records,
        "global": {"method": method_global, "shap": shap_global.tolist(), "lime": lime_global.tolist(),
                   "surrogate": surrogate.feature_importances_.tolist(),
                   "label_tree": label_tree.feature_importances_.tolist(), "permutation": perm.tolist()},
        "cost": {"method_s": float(np.mean(timings["method"])),
                 "shap_baseline_s": float(np.mean(timings["shap_baseline"])),
                 "shap_s_per_sample": float(t_shap),
                 "lime_s": float(np.mean(timings["lime"])), "lime_evals": args.lime_samples,
                 "method_evals": F + 2, "shap_baseline_evals": 2 ** F,
                 "shap_evals_per_sample": args.max_evals * args.background},
        "surrogate_top": export_text(surrogate, feature_names=feature_names(DATASET), max_depth=3),
    }


# --------------------------------------------------------------------------
# relatorio
# --------------------------------------------------------------------------


def pm(values, d: int = 3) -> str:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return f"{v.mean():.{d}f} ± {v.std():.{d}f}".replace(".", ",")


def fmt(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def local_tree_metrics(runs: list[dict], tree: str, block: str) -> dict:
    """Posicao do atributo em que a arvore separa i de j, em cada ranking, por semente."""
    keys = {"method": f"method@{block}", "floor": f"floor@{block}", **{k: k for k in RANKINGS
            if k not in ("method", "floor")}}
    out = {k: {"top1": [], "top3": [], "rr": []} for k in keys} | {"chance": {"top1": [], "top3": [], "rr": []}}
    counts = []
    for r in runs:
        per = {k: {"top1": [], "top3": [], "rr": []} for k in out}
        used = 0
        for p in r["pairs"]:
            t = p["trees"][tree]
            if not t["reproduces"] or t["feature"] is None:
                continue
            used += 1
            f, n = t["feature"], p["n_differ"]
            for k, key in keys.items():
                pos = rank_of(np.asarray(p["orders"][key]), f)
                per[k]["top1"].append(pos == 0)
                per[k]["top3"].append(pos < 3)
                per[k]["rr"].append(1.0 / (pos + 1))
            per["chance"]["top1"].append(1.0 / n)
            per["chance"]["top3"].append(min(3, n) / n)
            per["chance"]["rr"].append(sum(1.0 / (q + 1) for q in range(n)) / n)
        counts.append(used)
        for k in out:
            for m in out[k]:
                out[k][m].append(np.mean(per[k][m]) if per[k][m] else np.nan)
    return {"metrics": out, "used": counts}


def write_report(runs: list[dict], args, out: Path) -> None:
    feats, classes = feature_names(DATASET), class_names(DATASET)
    blocks, eb = runs[0]["blocks"], args.block
    n_pairs = sum(len(r["pairs"]) for r in runs)
    L = ["# Estudo de caso: `letter` — método × SHAP × árvore\n",
         f"{len(runs)} sementes, {args.pairs} pares por semente ({n_pairs} pares). MLP residual de "
         f"{args.blocks} blocos; método na Gram conjunta de {eb}, alvo fixo (classe {classes[args.fixed_target]}). "
         "Média ± desvio entre sementes.\n",
         "## Modelos\n",
         f"- MLP: acerto no teste {pm([r['accuracy'] for r in runs])}",
         f"- árvore substituta (ajustada às predições do MLP): fidelidade {pm([r['fidelity']['surrogate'] for r in runs])}, "
         f"acerto {pm([r['tree_accuracy']['surrogate'] for r in runs])}, "
         f"{pm([r['tree_leaves']['surrogate'] for r in runs], 0)} folhas, profundidade {pm([r['tree_depth']['surrogate'] for r in runs], 0)}",
         f"- árvore nos rótulos: acerto {pm([r['tree_accuracy']['label_tree'] for r in runs])}, "
         f"concordância com o MLP {pm([r['fidelity']['label_tree'] for r in runs])}",
         f"- atributos que diferem entre i e j: {pm([p['n_differ'] for r in runs for p in r['pairs']], 1)} de {runs[0]['n_features']}\n"]

    # 1. teste contrafactual
    keys = [f"method@{eb}", "shap_baseline", "shap", "shap_i", "lime", "lime_i", "absdiff", "gradxdelta", "random",
            f"floor@{eb}"]
    L += ["## 1. Teste contrafactual de trocas (área sob a curva, maior = melhor)\n",
          "| ranking | AUC | método vence em |", "|---|---|---|"]
    for k in keys:
        base = k.split("@")[0]
        wins = "—" if base == "method" else f"{np.mean([r['auc'][f'method@{eb}'] > r['auc'][k] for r in runs]):.0%}"
        L.append(f"| {LABEL[base]} | {pm([r['auc'][k] for r in runs])} | {wins} |")
    L += ["", "Método por bloco:\n", "| | " + " | ".join(blocks) + " |", "|---|" + "---|" * len(blocks)]
    for kind in ("method", "floor"):
        L.append(f"| {LABEL[kind]} | " + " | ".join(pm([r['auc'][f'{kind}@{b}'] for r in runs]) for b in blocks) + " |")

    sp = [p["spearman_method_shap"] for r in runs for p in r["pairs"] if "spearman_method_shap" in p]
    top1 = [np.argmax(p["method"][eb]) == np.argmax(p["shap_baseline"]) for r in runs for p in r["pairs"]]
    top3 = [len(set(np.argsort(-np.asarray(p["method"][eb]))[:3]) & set(np.argsort(-np.asarray(p["shap_baseline"]))[:3])) / 3
            for r in runs for p in r["pairs"]]
    L += ["", "Concordância método × Shapley exato, par a par (só atributos que diferem): "
          f"Spearman mediano {fmt(np.median(sp))} (quartis {fmt(np.quantile(sp, .25))} a {fmt(np.quantile(sp, .75))}); "
          f"mesmo 1º atributo em {np.mean(top1):.0%} dos pares; sobreposição média dos 3 primeiros {np.mean(top3):.0%}.\n"]

    c = runs[0]["cost"]
    L += ["Custo por par (GPU):\n", "| | avaliações do modelo | tempo |", "|---|---|---|",
          f"| método (todos os {len(blocks)} blocos) | {c['method_evals']} diretas + reversas | {fmt(np.mean([r['cost']['method_s'] for r in runs]) * 1e3, 1)} ms |",
          f"| Shapley exato | {c['shap_baseline_evals']} diretas | {fmt(np.mean([r['cost']['shap_baseline_s'] for r in runs]) * 1e3, 1)} ms |",
          f"| SHAP da biblioteca (2 amostras) | {2 * c['shap_evals_per_sample']} diretas | {fmt(2 * np.mean([r['cost']['shap_s_per_sample'] for r in runs]) * 1e3, 1)} ms |",
          f"| LIME contrastivo | {c['lime_evals']} diretas | {fmt(np.mean([r['cost']['lime_s'] for r in runs]) * 1e3, 1)} ms |", ""]

    # 2. arvore, par a par
    for tree, title in (("surrogate", "árvore substituta (predições do MLP)"), ("label_tree", "árvore nos rótulos")):
        res = local_tree_metrics(runs, tree, eb)
        m = res["metrics"]
        L += [f"## 2. Atributo em que a {title} separa i de j\n",
              f"Pares em que a árvore reproduz as duas predições do MLP: {pm(res['used'], 1)} de {args.pairs} por semente. "
              "Posição desse atributo em cada ranking.\n",
              "| ranking | 1º do ranking | entre os 3 primeiros | posto recíproco médio |", "|---|---|---|---|"]
        for k in ("method", "shap_baseline", "shap", "shap_i", "lime", "lime_i", "absdiff", "gradxdelta", "random",
                  "floor", "chance"):
            name = "acaso analítico entre os que diferem" if k == "chance" else LABEL[k]
            L.append(f"| {name} | {pm(m[k]['top1'])} | {pm(m[k]['top3'])} | {pm(m[k]['rr'])} |")
        L.append("")

    # 3. importancia global
    perm = np.array([r["global"]["permutation"] for r in runs])
    L += ["## 3. Importância global contra a permutação no MLP\n",
          "Spearman entre o vetor de importância de cada fonte e a queda de concordância do MLP ao permutar o atributo.\n",
          "| fonte | Spearman com a permutação |", "|---|---|"]
    gv = {}
    for k in GLOBAL:
        vals = [np.asarray(r["global"][k][eb] if k == "method" else r["global"][k]) for r in runs]
        gv[k] = np.array([normalized(v) for v in vals])
        L.append(f"| {GLOBAL_LABEL[k]} | {pm([spearmanr(v, p)[0] for v, p in zip(vals, perm)])} |")
    L += ["", "Método por bloco: " + ", ".join(
        f"{b} {pm([spearmanr(r['global']['method'][b], r['global']['permutation'])[0] for r in runs])}" for b in blocks), "",
          "Spearman entre o método e a árvore substituta: "
          f"{pm([spearmanr(r['global']['method'][eb], r['global']['surrogate'])[0] for r in runs])}; "
          f"entre o SHAP e a árvore substituta: {pm([spearmanr(r['global']['shap'], r['global']['surrogate'])[0] for r in runs])}.\n"]
    order = np.argsort(-perm.mean(0))
    L += ["Parcelas normalizadas (média entre sementes), na ordem da permutação:\n",
          "| atributo | significado | permutação | " + " | ".join(GLOBAL_LABEL[k] for k in GLOBAL) + " |",
          "|---|---|---|" + "---|" * len(GLOBAL)]
    pn = np.array([normalized(p) for p in perm]).mean(0)
    for f in order:
        L.append(f"| {feats[f]} | {DESCRIPTION[feats[f]]} | {fmt(pn[f])} | " + " | ".join(fmt(gv[k].mean(0)[f]) for k in GLOBAL) + " |")
    L += ["", "## Topo da árvore substituta (semente 0)\n", "```", runs[0]["surrogate_top"], "```"]
    (out / "tables.md").write_text("\n".join(L) + "\n")
    write_examples(runs[0], args, out, feats, classes)


def write_examples(run: dict, args, out: Path, feats, classes) -> None:
    """Pares da semente 0 lidos por extenso: método, Shapley exato, árvore e o método bloco a bloco."""
    eb, blocks = args.block, run["blocks"]
    E = ["# Exemplos de leitura — `letter`, semente 0\n",
         "Valores na escala original (inteiros de 0 a 15). Parcela = fração do ganho positivo de "
         f"proximidade (método, {eb}) ou do Shapley positivo (Shapley exato com j de referência).\n"]
    shown = 0
    for p in run["pairs"]:
        t = p["trees"]["surrogate"]
        if shown >= args.examples or not t["reproduces"] or t["feature"] is None:
            continue
        shown += 1
        m, s = np.asarray(p["method"][eb]), np.asarray(p["shap_baseline"])
        mo, so = np.argsort(-m), np.argsort(-s)
        f = t["feature"]
        side = lambda v: "≤" if v <= t["threshold"] else ">"  # noqa: E731
        E += [f"## {classes[p['pred_i']]} (i) × {classes[p['pred_j']]} (j)\n",
              f"i: rótulo {classes[p['true_i']]}, predita {classes[p['pred_i']]}; "
              f"j: rótulo {classes[p['true_j']]}, predita {classes[p['pred_j']]}; {p['n_differ']} atributos diferem.\n",
              f"Árvore substituta separa os dois em **{feats[f]}** ({DESCRIPTION[feats[f]]}) "
              f"no limiar {fmt(t['threshold'], 1)}: i = {p['raw_i'][f]:.0f} ({side(p['raw_i'][f])}), "
              f"j = {p['raw_j'][f]:.0f} ({side(p['raw_j'][f])}). "
              f"Posição no método: {rank_of(mo, f) + 1}º; no Shapley exato: {rank_of(so, f) + 1}º.\n",
              "| atributo | significado | i | j | parcela no método | parcela no Shapley exato |",
              "|---|---|---|---|---|---|"]
        mp, sp = normalized(m), normalized(s)
        for q in list(dict.fromkeys([*mo[:4], *so[:4]])):
            E.append(f"| {feats[q]} | {DESCRIPTION[feats[q]]} | {p['raw_i'][q]:.0f} | {p['raw_j'][q]:.0f} | "
                     f"{mp[q]:.0%} | {sp[q]:.0%} |")
        E += ["", "Parcela no método, bloco a bloco (o que o SHAP não separa):\n",
              "| atributo | " + " | ".join(blocks) + " |", "|---|" + "---|" * len(blocks)]
        for q in mo[:4]:
            E.append(f"| {feats[q]} | " + " | ".join(f"{normalized(p['method'][b])[q]:.0%}" for b in blocks) + " |")
        E.append("")
    (out / "exemplos.md").write_text("\n".join(E) + "\n")


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--block", default="b3")
    ap.add_argument("--steps", type=int, default=16, help="um passo por atributo")
    ap.add_argument("--background", type=int, default=100)
    ap.add_argument("--max-evals", type=int, default=1000)
    ap.add_argument("--global-samples", type=int, default=520)
    ap.add_argument("--lime-samples", type=int, default=5000, help="amostras de perturbação do LIME (padrão da biblioteca)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--min-leaf", type=int, default=3)
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
    ap.add_argument("--out", default="outputs/letter_case")
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
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, ensure_ascii=False))
        tmp.replace(path)
        runs.append(r)
        a = r["auc"]
        print(f"semente {seed} ({time.perf_counter() - t0:.0f} s): acerto {r['accuracy']:.3f}, "
              f"fidelidade {r['fidelity']['surrogate']:.3f} | AUC método {a[f'method@{args.block}']:.3f} "
              f"Shapley exato {a['shap_baseline']:.3f} SHAP {a['shap']:.3f} |Δx| {a['absdiff']:.3f} "
              f"piso {a[f'floor@{args.block}']:.3f}", flush=True)
    write_report(runs, args, out)
    from src.plots_case import plot_case_study
    plot_case_study(runs, args.block, LABEL, GLOBAL_LABEL, feature_names(DATASET), str(out / "case_study.png"))
    print(f"pronto: {out}/")


if __name__ == "__main__":
    main()

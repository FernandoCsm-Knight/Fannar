"""Bases, descricoes, regra de pares e utilitarios dos estudos de caso de P6.

Duas bases, escolhidas por `FANNAR_CASE`, nos dois lados da condicao que as vinte bases da suite
revelam (a vantagem da leitura de pares sobre |Δx| cresce com a acuracia do MLP):

  letter   decisao nitida (MLP com acuracia ~0,97). 16 atributos com significado (Frey & Slate, 1991),
           26 classes. Pares: i classificado corretamente e j o mais proximo com outra predicao.
  heloc    decisao fraca (nenhum modelo passa de ~0,73). HELOC (FICO, 2018), binario. Pergunta: "quais
           caracteristicas do historico financeiro precisariam ser diferentes para que o modelo
           classificasse este cliente como de menor risco?". Pares: i classificado corretamente como
           Bad, j o mais proximo classificado Good, a direcao em que um contrafactual e recurso.
"""

from __future__ import annotations

import os

import numpy as np
import torch

DATASET = os.environ.get("FANNAR_CASE", "letter")
assert DATASET in ("heloc", "letter"), f"FANNAR_CASE={DATASET}: os estudos de caso usam heloc ou letter"
BAD, GOOD = 0, 1  # classes do HELOC

# tamanho minimo das folhas da arvore de decisao, o de maior acuracia de validacao em cada base
# entre 1, 3, 10, 30, 50 e 100 (cinco sementes): na letter, a arvore crescida ate o fim; no HELOC,
# folhas menores decoram o ruido
MIN_LEAF = {"letter": 1, "heloc": 100}[DATASET]

DESCRIPTIONS = {
    # Dicionario de dados da FICO (Explainable Machine Learning Challenge, 2018)
    "heloc": {
        "ExternalRiskEstimate": "estimativa consolidada de risco de bureaus externos",
        "MSinceOldestTradeOpen": "meses desde a abertura da conta mais antiga",
        "MSinceMostRecentTradeOpen": "meses desde a abertura da conta mais recente",
        "AverageMInFile": "idade média das contas, em meses",
        "NumSatisfactoryTrades": "nº de contas em dia",
        "NumTrades60Ever2DerogPubRec": "nº de contas já com atraso de 60 dias ou mais",
        "NumTrades90Ever2DerogPubRec": "nº de contas já com atraso de 90 dias ou mais",
        "PercentTradesNeverDelq": "% de contas nunca em atraso",
        "MSinceMostRecentDelq": "meses desde o atraso mais recente",
        "MaxDelq2PublicRecLast12M": "pior atraso ou registro público nos últimos 12 meses (código)",
        "MaxDelqEver": "pior atraso já registrado (código)",
        "NumTotalTrades": "nº total de contas de crédito",
        "NumTradesOpeninLast12M": "nº de contas abertas nos últimos 12 meses",
        "PercentInstallTrades": "% de contas de parcelamento",
        "MSinceMostRecentInqexcl7days": "meses desde a consulta de crédito mais recente",
        "NumInqLast6M": "nº de consultas de crédito nos últimos 6 meses",
        "NumInqLast6Mexcl7days": "nº de consultas nos últimos 6 meses, exceto os últimos 7 dias",
        "NetFractionRevolvingBurden": "uso do crédito rotativo (saldo sobre limite, %)",
        "NetFractionInstallBurden": "saldo dos parcelamentos sobre o valor original (%)",
        "NumRevolvingTradesWBalance": "nº de contas rotativas com saldo",
        "NumInstallTradesWBalance": "nº de parcelamentos com saldo",
        "NumBank2NatlTradesWHighUtilization": "nº de cartões bancários com utilização alta",
        "PercentTradesWBalance": "% de contas com saldo",
    },
    # Frey & Slate (1991), "Letter recognition using Holland-style adaptive classifiers"
    "letter": {
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
    },
}
# Atributos que a pessoa nao pode mudar por uma acao propria: no HELOC, um escore externo, que e
# resultado e nao comportamento, e os que so dependem do tempo de historico. Na letter, nenhum.
IMMUTABLES = {
    "heloc": ("ExternalRiskEstimate", "MSinceOldestTradeOpen", "MSinceMostRecentTradeOpen", "AverageMInFile"),
    "letter": (),
}
DESCRIPTION, IMMUTABLE = DESCRIPTIONS[DATASET], IMMUTABLES[DATASET]


def mutable_indices(names: list[str]) -> np.ndarray:
    return np.array([k for k, n in enumerate(names) if n not in IMMUTABLE])


def source_mask(preds: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Quem pode ser i: classificado corretamente e numa classe de que faz sentido querer sair."""
    if DATASET == "heloc":
        return (preds == BAD) & (y == BAD)
    return preds == y


def target_classes(ci: int) -> np.ndarray:
    """Classes aceitas para j, dado c_i: Good no HELOC; qualquer outra classe na letter."""
    return np.array([GOOD]) if DATASET == "heloc" else np.setdiff1d(np.arange(26), [ci])


def select_pairs(preds: np.ndarray, y: np.ndarray, X: np.ndarray, rng, n: int) -> list[tuple[int, int]]:
    """i sorteado entre os de `source_mask`; j o mais proximo de i em X entre os preditos em `target_classes`."""
    pairs = []
    for i in rng.choice(np.flatnonzero(source_mask(preds, y)), size=n, replace=False):
        cand = np.flatnonzero(np.isin(preds, target_classes(int(preds[i]))))
        pairs.append((int(i), int(cand[np.argmin(np.linalg.norm(X[cand] - X[i], axis=1))])))
    return pairs


# --------------------------------------------------------------------------
# formatacao
# --------------------------------------------------------------------------


def pm(values, d: int = 3) -> str:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return f"{v.mean():.{d}f} ± {v.std():.{d}f}".replace(".", ",") if len(v) else "—"


def fmt(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def normalized(v: np.ndarray) -> np.ndarray:
    v = np.maximum(np.asarray(v, dtype=float), 0)
    return v / max(v.sum(), 1e-12)


def rank_of(order: np.ndarray, feature: int) -> int:
    return int(np.flatnonzero(np.asarray(order) == feature)[0])


def order_of(score: np.ndarray, rng) -> np.ndarray:
    """Decrescente, com empates desfeitos ao acaso (lexsort ordena pela ultima chave)."""
    return np.lexsort((rng.random(len(score)), -np.asarray(score)))


# --------------------------------------------------------------------------
# arvores
# --------------------------------------------------------------------------


def split_node(tree, a: np.ndarray, b: np.ndarray):
    """No em que os caminhos de a e b se separam (None se caem na mesma folha).

    O sklearn numera os nos em pre-ordem, entao todo ancestral tem indice menor: o ancestral
    comum mais profundo e o maior indice comum aos dois caminhos.
    """
    path = tree.decision_path(np.stack([a, b]).astype(np.float32))
    pa, pb = path[0].indices, path[1].indices
    node = int(np.intersect1d(pa, pb).max())
    if node == pa[-1] and node == pb[-1]:
        return None
    return node


def tree_baseline_shapley(tree, xi, xj, ci: int, cj: int) -> tuple[np.ndarray, float]:
    """Shapley exato de v(S) = [P(c_j) − P(c_i)](x_j em S, x_i fora), pelo TreeSHAP intervencional.

    Com um unico ponto de fundo (x_i) e explicando x_j, o TreeSHAP intervencional e exatamente o
    baseline Shapley do jogo acima, sem enumerar os 2^F subconjuntos.
    """
    import shap

    explainer = shap.TreeExplainer(tree, data=np.asarray(xi, dtype=np.float64)[None],
                                   feature_perturbation="interventional")
    sv = np.asarray(explainer.shap_values(np.asarray(xj, dtype=np.float64)[None], check_additivity=False))
    col = {c: k for k, c in enumerate(tree.classes_)}
    if sv.ndim == 3 and sv.shape[0] == 1:   # (1, F, K)
        sv = sv[0].T
    elif sv.ndim == 3:                      # (K, 1, F)
        sv = sv[:, 0, :]
    phi = sv[col[cj]] - sv[col[ci]]
    p = tree.predict_proba(np.stack([xi, xj]).astype(np.float32))
    total = float((p[1, col[cj]] - p[1, col[ci]]) - (p[0, col[cj]] - p[0, col[ci]]))
    return phi.astype(np.float64), total


def tree_min_flip(tree, a: np.ndarray, b: np.ndarray, target: int, allowed=None) -> int | None:
    """Menor numero de atributos levados de a a b (so os de `allowed`) que faz a arvore prever `target`.

    Percorre a arvore guardando quais atributos ja foram fixados em a ou em b: num no em que a e b
    seguem ramos diferentes, os dois ramos sao possiveis (o atributo fica com a ou passa a b). E exato,
    como a enumeracao dos 2^F subconjuntos, mas so visita os caminhos alcancaveis.
    """
    t = tree.tree_
    a32, b32 = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    allowed = set(range(len(a))) if allowed is None else {int(f) for f in allowed}
    target_col = int(np.flatnonzero(tree.classes_ == target)[0])
    best = [None]

    def visit(node: int, to_b: frozenset, to_a: frozenset) -> None:
        if best[0] is not None and len(to_b) >= best[0]:
            return
        if t.children_left[node] == -1:
            if int(np.argmax(t.value[node, 0])) == target_col:
                best[0] = len(to_b)
            return
        f, thr = int(t.feature[node]), t.threshold[node]
        left_a, left_b = a32[f] <= thr, b32[f] <= thr
        go = lambda left: t.children_left[node] if left else t.children_right[node]  # noqa: E731
        if f in to_b:
            visit(go(left_b), to_b, to_a)
        elif f in to_a or left_a == left_b or f not in allowed:
            visit(go(left_a), to_b, to_a | {f} if left_a != left_b else to_a)
        else:
            visit(go(left_b), to_b | {f}, to_a)
            visit(go(left_a), to_b, to_a | {f})

    visit(0, frozenset(), frozenset())
    return best[0]


# --------------------------------------------------------------------------
# modelo
# --------------------------------------------------------------------------


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

"""Leituras que ligam a geometria da Gram ao comportamento da rede.

A afirmacao em teste nao e "a geometria separa as classes" e sim "a geometria reflete o que
*este* modelo faz"; por isso a referencia e sempre algo que a rede produziu:

  knn_loo(d, preds)      a geometria recupera a predicao
  correctness_auc        a margem geometrica antecipa onde a rede erra
  confusion_agreement    as classes que o modelo confunde estao proximas
  cka                    alinhamento centrado entre duas Grams
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .geometry import mean_class_distance


def knn_loo(d: np.ndarray, target: np.ndarray, k: int = 5) -> float:
    """Acuracia kNN leave-one-out lida direto da matriz de distancia.

    Invariante a escala da distancia, ao contrario das medias, e por isso
    comparavel entre camadas e entre representacoes.
    """
    n = len(d)
    dd = d.copy()
    np.fill_diagonal(dd, np.inf)
    order = np.argsort(dd, axis=1)[:, :k]
    votes = target[order]
    hits = 0
    for i in range(n):
        vals, counts = np.unique(votes[i], return_counts=True)
        hits += int(vals[np.argmax(counts)] == target[i])
    return hits / n


def _center(k: np.ndarray) -> np.ndarray:
    n = len(k)
    h = np.eye(n) - np.ones((n, n)) / n
    return h @ k @ h


def cka(a: np.ndarray, b: np.ndarray) -> float:
    ac, bc = _center(a), _center(b)
    num = float((ac * bc).sum())
    den = float(np.linalg.norm(ac, "fro") * np.linalg.norm(bc, "fro"))
    return num / den if den > 0 else 0.0


def class_margin_against(
    d: np.ndarray, labels: np.ndarray, against: np.ndarray, n_classes: int = 10
) -> np.ndarray:
    """Margem geometrica por amostra contra um rotulo de referencia, em [-1, 1].

    (distancia media a classe competidora mais proxima - distancia media a classe
    de referencia) / soma: invariante a escala de d, positiva quando a amostra
    esta dentro da regiao da classe contra a qual e comparada.
    """
    means = mean_class_distance(d, labels, n_classes)
    rows = np.arange(len(d))
    own = means[rows, against]
    other = means.copy()
    other[rows, against] = np.inf
    near = other.min(1)
    return (near - own) / np.maximum(near + own, 1e-12)


def correctness_auc(
    d: np.ndarray, labels: np.ndarray, preds: np.ndarray, n_classes: int = 10
) -> float:
    """AUC da margem geometrica (contra o rotulo) prevendo o acerto da rede.

    0,5 e o acaso. Le-se: a geometria sabe, sem olhar os logits, em quais imagens
    a rede vai errar.
    """
    geo = class_margin_against(d, labels, labels, n_classes)
    correct = preds == labels
    pos, neg = geo[correct], geo[~correct]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def class_distance_matrix(d: np.ndarray, labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """(K, K) distancia media entre as classes, a geometria no nivel de classe."""
    out = np.zeros((n_classes, n_classes))
    for a in range(n_classes):
        ia = labels == a
        for b in range(n_classes):
            ib = labels == b
            if not ia.any() or not ib.any():
                out[a, b] = np.nan
                continue
            block = d[np.ix_(ia, ib)]
            if a == b:
                m = ~np.eye(block.shape[0], dtype=bool)
                out[a, b] = block[m].mean() if m.any() else 0.0
            else:
                out[a, b] = block.mean()
    return out


def confusion_agreement(
    d: np.ndarray, labels: np.ndarray, preds: np.ndarray, n_classes: int = 10
) -> tuple[float, float]:
    """Spearman entre a proximidade geometrica das classes e a confusao do modelo.

    Compara os pares fora da diagonal de $-M_{ab}$ (proximidade) com a confusao
    simetrizada $c_{ab} + c_{ba}$. Positivo significa que os pares que o modelo
    troca sao os pares que o espaco de representacao aproxima.
    """
    m = class_distance_matrix(d, labels, n_classes)
    conf = np.zeros((n_classes, n_classes))
    for t, p in zip(labels, preds):
        conf[t, p] += 1
    conf = conf + conf.T
    iu = np.triu_indices(n_classes, k=1)
    rho, p = spearmanr(-m[iu], conf[iu])
    return float(rho), float(p)



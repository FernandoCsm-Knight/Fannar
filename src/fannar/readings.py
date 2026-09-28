"""Readings that relate a representation space to the behaviour of the model.

The reference is always something the *model* produced (its predictions, its errors, its
confusion), because the claim under test is that the geometry reflects what this model does,
not that it separates the dataset's classes.

=====================  =====================================================================
``knn_prediction``      leave-one-out k-NN accuracy against the model's predictions
``correctness_auc``     AUC of the geometric margin as a predictor of the model being right
``confusion_agreement`` Spearman between class proximity and the model's (soft) confusion
``containment``         class directions inside the retained subspace (r = K - 1)
``e_ratio_error_auc``   AUC of the orthogonal-energy ratio as a predictor of an error
=====================  =====================================================================
"""

from __future__ import annotations

import numpy as np

from .space import GramSpace


def auc(score, positive) -> float:
    """Area under the ROC curve of ``score`` for the boolean ``positive`` (Mann-Whitney)."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = _rankdata(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), 1-based."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / counts)[inv]


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(_rankdata(a), _rankdata(b))[0, 1])


def _distances(space) -> np.ndarray:
    return space.distances if isinstance(space, GramSpace) else np.asarray(space, float)


def knn_prediction(space, predictions, k: int = 5) -> float:
    """Leave-one-out k-NN accuracy of the space against the model's own predictions. O(n^2)."""
    d = _distances(space).copy()
    target = np.asarray(predictions)
    np.fill_diagonal(d, np.inf)
    k = min(k, len(d) - 1)
    nn = np.argpartition(d, k - 1, axis=1)[:, :k]
    votes = target[nn]
    hits = 0
    for i in range(len(d)):
        vals, counts = np.unique(votes[i], return_counts=True)
        hits += int(vals[np.argmax(counts)] == target[i])
    return hits / len(d)


def class_distance_matrix(space, labels) -> tuple[np.ndarray, np.ndarray]:
    """``(K, K)`` mean distance between the objects of each pair of classes, and the classes."""
    d, labels = _distances(space), np.asarray(labels)
    classes = np.unique(labels)
    out = np.full((len(classes), len(classes)), np.nan)
    for a, ca in enumerate(classes):
        ia = labels == ca
        for b, cb in enumerate(classes):
            block = d[np.ix_(ia, labels == cb)]
            if a == b:
                m = ~np.eye(len(block), dtype=bool)
                out[a, b] = block[m].mean() if m.any() else 0.0
            else:
                out[a, b] = block.mean()
    return out, classes


def geometric_margin(space, labels, against=None) -> np.ndarray:
    """Per object, ``(d_near - d_own) / (d_near + d_own)`` in ``[-1, 1]``: mean distance to the
    closest competing class minus mean distance to the reference class (``against``, default the
    object's own label), over their sum. Positive inside the reference class's region."""
    d, labels = _distances(space), np.asarray(labels)
    classes = np.unique(labels)
    idx = {c: i for i, c in enumerate(classes)}
    d = d.copy()
    np.fill_diagonal(d, np.nan)  # an object's distance to itself is not evidence of anything
    with np.errstate(invalid="ignore"):
        means = np.stack([np.nanmean(d[:, labels == c], axis=1) for c in classes], axis=1)
    means = np.nan_to_num(means, nan=0.0)
    own_cls = np.array([idx[c] for c in (labels if against is None else np.asarray(against))])
    rows = np.arange(len(d))
    own = means[rows, own_cls]
    other = means.copy()
    other[rows, own_cls] = np.inf
    near = other.min(1)
    return (near - own) / np.maximum(near + own, 1e-12)


def correctness_auc(space, labels, predictions) -> float:
    """How well the geometric margin (against the true label) predicts that the model is right."""
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    return auc(geometric_margin(space, labels), predictions == labels)


def soft_confusion(probabilities, labels) -> tuple[np.ndarray, np.ndarray]:
    """``S_ab`` = mean probability the model gives to class ``b`` on objects of class ``a``,
    symmetrised. Informative even when the model rarely errs."""
    p, labels = np.asarray(probabilities, float), np.asarray(labels)
    classes = np.unique(labels)
    s = np.stack([p[labels == c][:, classes].mean(0) for c in classes])
    return s + s.T, classes


def count_confusion(labels, predictions) -> tuple[np.ndarray, np.ndarray]:
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    classes = np.unique(labels)
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)))
    for t, p in zip(labels, predictions):
        if p in idx:
            m[idx[t], idx[p]] += 1
    return m + m.T, classes


def confusion_agreement(space, labels, predictions=None, probabilities=None) -> float:
    """Spearman between the proximity of pairs of classes and the model's confusion between them
    (soft confusion if ``probabilities`` are given, counts otherwise). Needs >= 5 classes."""
    m, classes = class_distance_matrix(space, labels)
    if len(classes) < 5:
        return float("nan")
    if probabilities is not None:
        conf, _ = soft_confusion(probabilities, labels)
    else:
        conf, _ = count_confusion(labels, predictions)
    iu = np.triu_indices(len(classes), k=1)
    return spearman(-m[iu], conf[iu])


def e_ratio_error_auc(space: GramSpace, labels, predictions, r=None) -> float:
    """AUC of ``E_ratio`` (with ``r = K - 1`` by default) as a predictor of a model error."""
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    k = len(np.unique(labels)) - 1 if r is None else r
    return auc(space.energies(k)["E_ratio"], predictions != labels)


READINGS = ("knn_prediction", "correctness_auc", "confusion_agreement", "containment", "e_ratio_error_auc")


def evaluate(space: GramSpace, labels, predictions, probabilities=None, k: int = 5,
             min_errors: int = 5) -> dict:
    """All behaviour readings of one space. Readings that need errors return NaN when the model
    makes fewer than ``min_errors`` errors on the sample."""
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    errors = int((predictions != labels).sum())
    enough = min_errors <= errors <= len(labels) - min_errors
    nan = float("nan")
    return {
        "knn_prediction": knn_prediction(space, predictions, k),
        "correctness_auc": correctness_auc(space, labels, predictions) if enough else nan,
        "confusion_agreement": confusion_agreement(space, labels, predictions, probabilities),
        "containment": space.containment(labels),
        "e_ratio_error_auc": e_ratio_error_auc(space, labels, predictions) if enough else nan,
        "rank": space.participation_rank(),
    }

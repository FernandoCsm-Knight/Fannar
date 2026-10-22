"""Why was this object classified differently from that one? (contrastive, per feature)

For a pair ``(i, j)`` and each feature ``f``, ``x_i^(f)`` is ``x_i`` with feature ``f`` replaced by
its value in ``x_j``, and

    Delta d_f = d(i, j) - d(i^(f), j)

in the representation space of each layer (positive: the swap brings ``i`` closer to ``j``).
Only the ``F + 2`` rows ``x_j, x_i, x_i^(1..F)`` go through the model, so the cost is linear in
the number of features. The ranking is validated outside the geometry by the swap test: copy
the top-``k`` features of ``j`` into ``i`` and check whether the model's prediction becomes the
prediction of ``j`` (the procedure of nearest-instance counterfactuals such as NICE).

Delta d is first order: features that only matter jointly with others get no credit.

:class:`ErrorExplanation` aggregates the same scores over the model's errors: each misclassified
object is paired with the nearest object the model gets right in its true class, so the shares
say which features keep the errors away from where they belong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PairExplanation:
    """Per-layer feature scores for one pair."""

    i: int | None
    j: int | None
    xi: np.ndarray
    xj: np.ndarray
    scores: dict[str, np.ndarray]
    pred_i: int | None = None
    pred_j: int | None = None
    feature_names: list[str] | None = None
    meta: dict = field(default_factory=dict)

    @property
    def layers(self) -> list[str]:
        return list(self.scores)

    def ranking(self, layer: str | None = None) -> np.ndarray:
        """Feature indices, most distinguishing first."""
        s = self.scores[layer or self.layers[-1]]
        return np.argsort(-s, kind="stable")

    def shares(self, layer: str | None = None) -> np.ndarray:
        """Positive part of the scores, normalised to sum 1 (share of the gain in proximity)."""
        s = np.maximum(self.scores[layer or self.layers[-1]], 0)
        return s / s.sum() if s.sum() > 0 else s

    def top(self, k: int = 5, layer: str | None = None) -> list[tuple[str, float, float, float]]:
        """``[(feature, value_i, value_j, share), ...]`` for the top ``k`` features."""
        names = self.feature_names or [f"x{t}" for t in range(len(self.xi))]
        sh = self.shares(layer)
        return [(names[f], float(self.xi[f]), float(self.xj[f]), float(sh[f])) for f in self.ranking(layer)[:k]]

    def to_frame(self, layer: str | None = None):
        import pandas as pd
        names = self.feature_names or [f"x{t}" for t in range(len(self.xi))]
        return pd.DataFrame({"feature": names, "value_i": self.xi, "value_j": self.xj,
                             "score": self.scores[layer or self.layers[-1]], "share": self.shares(layer)}
                            ).sort_values("score", ascending=False, ignore_index=True)

    def plot(self, k: int = 8):
        from .plots import plot_pair
        return plot_pair(self, k)


@dataclass
class ErrorExplanation:
    """Per-layer feature shares over the model's errors, one row per error.

    ``partners[e]`` is the object error ``e`` was contrasted with (by default, the nearest object
    correctly predicted as its true class); ``shares[layer][e, f]`` is the share of feature ``f``
    in bringing error ``e`` closer to it in the representation space. ``xi`` and ``xj`` hold the
    feature values of the errors and of their partners. ``reference`` holds the same shares over
    correctly predicted objects (each against its nearest unlike object), i.e. how much each
    feature matters to the model in general; ``support`` counts the objects of each true class.
    """

    errors: np.ndarray
    partners: np.ndarray
    true: np.ndarray
    pred: np.ndarray
    shares: dict[str, np.ndarray]
    xi: np.ndarray
    xj: np.ndarray
    reference: dict[str, np.ndarray] = field(default_factory=dict)
    support: dict[int, int] = field(default_factory=dict)
    feature_names: list[str] | None = None
    class_names: list[str] | None = None

    @property
    def layers(self) -> list[str]:
        return list(self.shares)

    @property
    def names(self) -> list[str]:
        return self.feature_names or [f"x{t}" for t in range(self.xi.shape[1])]

    @property
    def pairs(self) -> list[tuple[int, int]]:
        """``(error, partner)`` pairs, e.g. for :meth:`Interpreter.swap_test`."""
        return [(int(i), int(j)) for i, j in zip(self.errors, self.partners)]

    def mean_shares(self, layer: str | None = None, mask=None) -> np.ndarray:
        """Mean share of each feature over the errors (or over the ``mask`` of them)."""
        s = self.shares[layer or self.layers[-1]]
        return (s if mask is None else s[mask]).mean(0)

    def reference_shares(self, layer: str | None = None) -> np.ndarray | None:
        """Mean share of each feature over the correctly predicted objects, if computed."""
        s = self.reference.get(layer or self.layers[-1])
        return None if s is None or len(s) == 0 else s.mean(0)

    def direction(self, layer: str | None = None, mask=None) -> np.ndarray:
        """Share-weighted mean of ``sign(x_j - x_i)`` per feature, in ``[-1, 1]``: ``+1`` when the
        errors have lower values than their partners (raising the feature moves them towards
        their true class), ``-1`` when they have higher values."""
        s, d = self.shares[layer or self.layers[-1]], np.sign(self.xj - self.xi)
        if mask is not None:
            s, d = s[mask], d[mask]
        w = s.sum(0)
        return np.where(w > 0, (s * d).sum(0) / np.maximum(w, 1e-300), 0.0)

    def top_fraction(self, layer: str | None = None) -> np.ndarray:
        """Fraction of the errors in which each feature is the most distinguishing one."""
        s = self.shares[layer or self.layers[-1]]
        return np.bincount(s.argmax(1), minlength=s.shape[1]) / max(len(s), 1)

    def ranking(self, layer: str | None = None) -> np.ndarray:
        """Feature indices, most responsible for the errors first."""
        return np.argsort(-self.mean_shares(layer), kind="stable")

    def top(self, k: int = 5, layer: str | None = None) -> list[tuple[str, float, float]]:
        """``[(feature, mean share, top fraction), ...]`` for the top ``k`` features."""
        mean, first = self.mean_shares(layer), self.top_fraction(layer)
        return [(self.names[f], float(mean[f]), float(first[f])) for f in self.ranking(layer)[:k]]

    def confusions(self) -> list[tuple[int, int, int]]:
        """``(true, predicted, count)`` among the errors, most frequent first."""
        pairs, counts = np.unique(np.stack([self.true, self.pred], 1), axis=0, return_counts=True)
        return [(int(pairs[t, 0]), int(pairs[t, 1]), int(counts[t])) for t in np.argsort(-counts, kind="stable")]

    def by_class(self, layer: str | None = None) -> dict[int, np.ndarray]:
        """Mean shares per true class of the errors."""
        return {int(c): self.mean_shares(layer, self.true == c) for c in np.unique(self.true)}

    def by_confusion(self, layer: str | None = None) -> dict[tuple[int, int], np.ndarray]:
        """Mean shares per ``(true, predicted)`` confusion."""
        return {(a, b): self.mean_shares(layer, (self.true == a) & (self.pred == b)) for a, b, _ in self.confusions()}

    def to_frame(self, layer: str | None = None):
        import pandas as pd
        ref = self.reference_shares(layer)
        return pd.DataFrame({"feature": self.names, "mean_share": self.mean_shares(layer),
                             "reference_share": np.full(len(self.names), np.nan) if ref is None else ref,
                             "top_fraction": self.top_fraction(layer), "direction": self.direction(layer)}
                            ).sort_values("mean_share", ascending=False, ignore_index=True)

    def plot(self, k: int | None = None, layer: str | None = None, rows: str = "class"):
        from .plots import plot_errors
        return plot_errors(self, k, layer, rows)


def nearest_correct(X, labels, predictions, i: int) -> int | None:
    """The nearest object (Euclidean, in the given feature space) correctly predicted as the
    true class of ``i``, or ``None`` if there is none."""
    X, labels, predictions = np.asarray(X, float).reshape(len(X), -1), np.asarray(labels), np.asarray(predictions)
    pool = np.flatnonzero((predictions == labels) & (labels == labels[i]))
    if len(pool) == 0:
        return None
    return int(pool[np.argmin(np.linalg.norm(X[pool] - X[i], axis=1))])


def swap_rows(xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
    """``[x_j, x_i, x_i^(1), ..., x_i^(F)]``."""
    F = len(xi)
    swapped = np.repeat(xi[None], F, axis=0)
    swapped[np.arange(F), np.arange(F)] = xj
    return np.concatenate([xj[None], xi[None], swapped])


def pair_scores(extractor, pipeline, xi, xj, layers=None) -> dict[str, np.ndarray]:
    """``Delta d_f`` per layer, from the ``F + 2`` swap rows."""
    xi, xj = np.asarray(xi, float).ravel(), np.asarray(xj, float).ravel()
    src = extractor(swap_rows(xi, xj).astype(np.float32))
    out = {}
    for layer in layers or src.layer_names:
        sources = src.layers[layer]
        if all(np.array_equal(a, np.broadcast_to(a[:1], a.shape)) for a in sources.values()):
            out[layer] = np.zeros(len(xi))  # every row has the same sources: nothing separates i from j here
            continue
        d = pipeline(sources=sources, keep_components=False).distances[0]
        out[layer] = d[1] - d[2:]
    return out


def swap_curve(predict, xi, xj, target: int, order) -> np.ndarray:
    """For ``k = 0..F``: 1 if ``x_i`` with the first ``k`` features of ``order`` taken from
    ``x_j`` is predicted as ``target``. ``predict(X) -> labels``."""
    xi, xj = np.asarray(xi, float).ravel(), np.asarray(xj, float).ravel()
    F = len(xi)
    rank = np.empty(F, dtype=int)
    rank[np.asarray(order)] = np.arange(F)
    rows = np.repeat(xi[None], F + 1, axis=0)
    for k in range(F + 1):
        rows[k, rank < k] = xj[rank < k]
    return (np.asarray(predict(rows.astype(np.float32))) == target).astype(float)


def curve_area(curve: np.ndarray) -> float:
    """Area under a swap curve on ``[0, 1]`` (fraction of features swapped)."""
    x = np.linspace(0, 1, len(curve))
    return float(((curve[1:] + curve[:-1]) * 0.5 * np.diff(x)).sum())


def nearest_unlike(X, predictions, i: int) -> int:
    """The nearest object (Euclidean, in the given feature space) with another prediction."""
    X, predictions = np.asarray(X, float).reshape(len(X), -1), np.asarray(predictions)
    other = np.flatnonzero(predictions != predictions[i])
    if len(other) == 0:
        raise ValueError("every object has the same prediction")
    return int(other[np.argmin(np.linalg.norm(X[other] - X[i], axis=1))])

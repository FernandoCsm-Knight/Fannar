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
        d = pipeline(sources=src.layers[layer], keep_components=False).distances[0]
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

"""Sources of a scikit-learn decision tree, one "layer" per depth.

A tree has no activations and no gradient, so it needs another instantiation of the sources;
the construction of the method is unchanged. At depth ``l`` each object has:

* ``path``    -- indicator of the nodes visited up to depth ``l`` (the state of the tree);
* ``slack``   -- ``(x_f - t) / sigma_f`` at every internal node already crossed: how far the
  object is from switching branch (the tree's sensitivity);
* ``classes`` -- class distribution of the node the object reaches at depth ``l``.
"""

from __future__ import annotations

import numpy as np

from . import Sources

TREE_SOURCES = ("path", "slack", "classes")


class TreeSources:
    """Per-depth sources of a fitted ``DecisionTreeClassifier`` (or any sklearn tree with ``tree_``).

    Parameters
    ----------
    tree:
        Fitted scikit-learn decision tree classifier.
    scale:
        Per-feature scale for the slack (e.g. the training standard deviation); defaults to 1.
    depths:
        Depths to use as layers; defaults to ``1 .. max_depth``.
    """

    def __init__(self, tree, scale=None, depths=None) -> None:
        t = tree.tree_
        self.tree = tree
        self.internal = t.children_left != -1
        self.feature = np.where(self.internal, t.feature, 0)
        self.threshold = t.threshold
        value = t.value[:, 0, :]
        self.value = value / np.maximum(value.sum(1, keepdims=True), 1e-300)
        self.depth = np.zeros(t.node_count, dtype=int)
        for n in range(t.node_count):  # pre-order: parents before children
            for child in (t.children_left[n], t.children_right[n]):
                if child != -1:
                    self.depth[child] = self.depth[n] + 1
        self.max_depth = int(self.depth.max())
        self.scale = None if scale is None else np.asarray(scale, float)
        self.depths = list(depths) if depths is not None else list(range(1, self.max_depth + 1))

    def at_depth(self, X, depth: int) -> dict[str, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        scale = np.ones(X.shape[1]) if self.scale is None else self.scale
        P = self.tree.decision_path(X.astype(np.float32)).toarray().astype(np.float64)
        path = P * (self.depth <= depth)
        crossed = P * (self.depth < depth) * self.internal
        slack = crossed * (X[:, self.feature] - self.threshold) / scale[self.feature]
        node = (path * np.arange(path.shape[1])).argmax(1)  # deepest node of the prefix
        return {"path": path, "slack": slack, "classes": self.value[node]}

    def __call__(self, X) -> Sources:
        layers = {f"depth{d}": self.at_depth(X, d) for d in self.depths}
        proba = self.tree.predict_proba(np.asarray(X, dtype=np.float32))
        preds = self.tree.classes_[proba.argmax(1)]
        return Sources(layers, preds, proba, None, {"kind": "tree"})

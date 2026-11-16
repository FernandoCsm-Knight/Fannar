"""Decision trees: a fitted scikit-learn tree, one "layer" per depth.

>>> from fannar.tree import TreeInterpreter
>>> interp = TreeInterpreter(tree, df_test, y_test, X_train=df_train)   # slack in training std units
>>> interp.evaluate()
>>> interp.explain_pair(0).top(5)

A tree has no activations and no gradient. Its sources are ``path``, ``slack`` and ``classes`` at
every depth (see :class:`~fannar.TreeSources`). The construction of the space is the same as for a
network. Here the joint pipeline is ``path + slack + classes`` and the baseline is ``path`` alone.

Labels can be the tree's own classes (strings included). They are mapped to indices into
``tree.classes_``, which are also the class names, so every reading and explanation works with
any label type.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .interpreter import Interpreter
from .sources import Sources
from .sources.tree import TREE_SOURCES, TreeSources
from .tabular import as_table


class _IndexedTreeSources(TreeSources):
    """:class:`TreeSources` whose predictions are indices into ``tree.classes_``."""

    def __call__(self, X) -> Sources:
        src = super().__call__(X)
        src.predictions = src.probabilities.argmax(1)
        return src


class TreeInterpreter(Interpreter):
    """:class:`~fannar.Interpreter` for a fitted scikit-learn decision tree classifier.

    Parameters
    ----------
    tree:
        A fitted ``DecisionTreeClassifier`` (anything with ``tree_``, ``decision_path`` and
        ``predict_proba``).
    X:
        Rows to interpret: a ``(n, F)`` array or a ``pandas.DataFrame`` (its columns become the
        feature names).
    y:
        True labels, with the same values the tree was trained on (``tree.classes_``).
    X_train:
        Optional training rows. Their per-feature standard deviation becomes the scale of the
        ``slack`` source, so distances to the thresholds are comparable across features.
    scale:
        An explicit per-feature scale for the slack (overrides ``X_train``).
    depths:
        Depths used as layers (default ``1 .. max_depth``).
    **kwargs:
        Passed to :class:`~fannar.Interpreter` (``pipeline``, ``baselines``, ``reference_model``, ``k``).
    """

    def __init__(self, tree, X, y=None, *, X_train=None, scale=None, depths: Sequence[int] | None = None,
                 feature_names: Sequence[str] | None = None, class_names: Sequence[str] | None = None,
                 **kwargs) -> None:
        classes = getattr(tree, "classes_", None)
        X, y, feature_names, class_names = as_table(X, y, feature_names, class_names, classes=classes,
                                                    dtype=np.float64)
        if class_names is None and classes is not None and y is not None and not np.array_equal(
                np.asarray(classes), np.arange(len(classes))):
            class_names = [str(c) for c in classes]
        if scale is None and X_train is not None:
            Xt = np.asarray(X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train, dtype=np.float64)
            scale = np.where(Xt.std(0) > 0, Xt.std(0), 1.0)
        reference = kwargs.pop("reference_model", None)
        extractor = _IndexedTreeSources(tree, scale=scale, depths=depths)
        super().__init__(tree, X, y, extractor=extractor, feature_names=feature_names,
                         class_names=class_names, **kwargs)
        if reference is not None:  # another tree (e.g. a shallower or randomised one) as the floor
            self.reference_extractor = _IndexedTreeSources(reference, scale=scale, depths=depths)

    def _predict_fn(self):
        tree = self.model
        return lambda rows: tree.predict_proba(np.asarray(rows, dtype=np.float32)).argmax(1)


__all__ = ["TreeInterpreter", "TreeSources", "TREE_SOURCES"]

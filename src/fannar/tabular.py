"""Tabular data: a PyTorch model over rows of features.

>>> from fannar.tabular import TabularInterpreter
>>> interp = TabularInterpreter(model, df_test, y_test, layers=["block1", "block2"],
...                             reference_model=untrained_copy)
>>> interp.evaluate().to_frame()
>>> interp.explain_errors().top(5)          # features named after df_test.columns

Only the input handling differs from :class:`~fannar.Interpreter`: a ``pandas.DataFrame`` gives
the feature names, a categorical or string ``y`` gives the class names, and the construction of
the representation space is unchanged. Nothing is trained or preprocessed here; pass the rows in
the scale the model was trained on.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .interpreter import Interpreter, require_torch


def encode_labels(y, classes: Sequence | None = None) -> tuple[np.ndarray | None, list[str] | None]:
    """``y`` as class indices ``0 .. K-1`` plus the class names, when ``y`` is not already numeric.

    A pandas categorical keeps its category order. Strings, objects and booleans are matched against
    ``classes`` when given (e.g. ``tree.classes_``), otherwise against their sorted unique values, the
    same order as scikit-learn's ``LabelEncoder``. Numeric labels are returned unchanged with no names.
    """
    if y is None:
        return None, None
    cat = getattr(y, "cat", None)
    if cat is not None and str(getattr(y, "dtype", "")) == "category":
        return np.asarray(cat.codes, dtype=np.int64), [str(c) for c in cat.categories]
    y = np.asarray(y)
    if classes is not None:
        classes = np.asarray(classes)
        if np.issubdtype(classes.dtype, np.integer) and np.array_equal(classes, np.arange(len(classes))):
            return y, None
        index = {c: t for t, c in enumerate(classes.tolist())}
        unknown = sorted({str(c) for c in y.tolist() if c not in index})
        if unknown:
            raise ValueError(f"labels {unknown[:5]} are not among the model's classes {classes.tolist()}")
        return np.array([index[c] for c in y.tolist()], dtype=np.int64), [str(c) for c in classes]
    if np.issubdtype(y.dtype, np.number):
        return y, None
    names, codes = np.unique(y, return_inverse=True)
    return codes.astype(np.int64), [str(c) for c in names]


def as_table(X, y=None, feature_names: Sequence[str] | None = None, class_names: Sequence[str] | None = None,
             classes: Sequence | None = None, dtype=np.float32):
    """``(X, y, feature_names, class_names)`` from arrays or pandas objects.

    Explicit ``feature_names``/``class_names`` take precedence over what the DataFrame and the labels
    provide. ``X`` becomes a ``dtype`` array (``float32`` by default, the usual dtype of a model).
    """
    if hasattr(X, "columns") and feature_names is None:
        feature_names = [str(c) for c in X.columns]
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    if not hasattr(X, "detach"):  # leave tensors to the model's own dtype handling
        X = np.asarray(X, dtype=dtype)
    y, names = encode_labels(y, classes)
    return X, y, feature_names, class_names if class_names is not None else names


class TabularInterpreter(Interpreter):
    """:class:`~fannar.Interpreter` for a PyTorch model over tabular rows.

    Parameters
    ----------
    model:
        A trained PyTorch classifier returning logits ``(N, K)``.
    X:
        Rows to interpret: a ``(n, F)`` array, tensor or ``pandas.DataFrame`` (its columns become
        the feature names).
    y:
        True labels: integers ``0 .. K-1`` in the model's output order, or a categorical/string
        ``pandas.Series`` or array (encoded as described in :func:`encode_labels`; its order must be
        the one used to train the model).
    layers:
        Module names to probe, as in ``model.named_modules()``; ``None`` probes every direct child
        with parameters.
    **kwargs:
        Everything else is passed to :class:`~fannar.Interpreter` (``reference_model``, ``pipeline``,
        ``baselines``, ``k``) and from it to :class:`~fannar.TorchSources` (``target``, ``fixed_class``,
        ``device``, ``batch_size``...).
    """

    def __init__(self, model, X, y=None, *, layers: Sequence[str] | None = None,
                 feature_names: Sequence[str] | None = None, class_names: Sequence[str] | None = None,
                 **kwargs) -> None:
        require_torch("fannar.tabular")
        X, y, feature_names, class_names = as_table(X, y, feature_names, class_names)
        super().__init__(model, X, y, layers=layers, feature_names=feature_names, class_names=class_names,
                         **kwargs)


__all__ = ["TabularInterpreter", "as_table", "encode_labels"]

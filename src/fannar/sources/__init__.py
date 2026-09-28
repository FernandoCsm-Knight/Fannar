"""Sources of information indexed by object -- the *instantiation* of the method.

The method needs, for each object and each layer, one array per source. Any extractor that
returns a :class:`Sources` works with :class:`~fannar.Interpreter`:

* :class:`TorchSources` -- activations, contrastive gradient and activated parameters of any
  PyTorch module (MLP, CNN, transformer);
* :class:`TreeSources` -- path, slack and class distribution per depth of a scikit-learn tree;
* :class:`FunctionSources` -- wrap your own ``fn(X) -> {layer: {source: array}}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Sources:
    """Per-layer sources for ``n`` objects and the model's behaviour on them.

    ``layers[layer][source]`` is an array whose first axis indexes the objects.
    """

    layers: dict[str, dict[str, np.ndarray]]
    predictions: np.ndarray
    probabilities: np.ndarray | None = None
    logits: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def layer_names(self) -> list[str]:
        return list(self.layers)

    def __len__(self) -> int:
        return len(self.predictions)


class FunctionSources:
    """Wraps ``fn(X) -> Sources`` or ``fn(X) -> (layers_dict, predictions[, probabilities])``."""

    def __init__(self, fn: Callable) -> None:
        self.fn = fn

    def __call__(self, X) -> Sources:
        out = self.fn(X)
        if isinstance(out, Sources):
            return out
        layers, preds, *rest = out
        return Sources(layers, np.asarray(preds), np.asarray(rest[0]) if rest else None)


from .tree import TreeSources  # noqa: E402

try:  # torch is optional
    from .torch import TorchSources  # noqa: E402
except ImportError:  # pragma: no cover
    TorchSources = None

__all__ = ["Sources", "FunctionSources", "TorchSources", "TreeSources"]

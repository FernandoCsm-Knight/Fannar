"""Composição sequencial de transformações pós-kernel."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..types import KernelTransform


class ComposeTransforms:
    """Aplica transformações em ordem: ``T_n(... T_1(K))``.

    Como a classe ``T_C`` é fechada por composição, o resultado permanece
    admissível.
    """

    name = "compose"

    def __init__(self, transforms: Sequence[KernelTransform]) -> None:
        self.transforms = list(transforms)

    def __call__(self, K: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            K = t(K)
        return K

    def __repr__(self) -> str:  # pragma: no cover
        inner = ", ".join(repr(t) for t in self.transforms)
        return f"ComposeTransforms([{inner}])"

"""Kernels par a par genéricos e a partir de matriz pré-computada."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .base import BaseKernel


class PrecomputedKernel(BaseKernel):
    """Envolve uma matriz de Gram já computada como um Kernel.

    Útil quando a similaridade vem de fora (ex.: pesos de atenção, matriz de
    correlação). ``x`` é ignorado; ``y`` deve ser ``None`` ou igual a ``x``.
    """

    name = "precomputed"

    def __init__(self, K: torch.Tensor) -> None:
        self.K = 0.5 * (K + K.T)

    def _compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return self.K

    def __call__(self, x=None, y=None):
        return self.K


class CallableKernel(BaseKernel):
    """Kernel definido por uma função par a par ``fn(x (n,d), y (m,d)) -> (n,m)``."""

    name = "callable"

    def __init__(self, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
        self.fn = fn

    def _compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.fn(x, y)

"""Kernel cosseno: ``k(x, y) = <x, y> / (||x|| ||y||)``."""

from __future__ import annotations

import torch

from .base import BaseKernel


class CosineKernel(BaseKernel):
    """Similaridade cosseno. Equivale ao kernel linear sobre vetores normalizados.

    Entradas com norma nula são tratadas via ``eps`` (resultado ~0 para elas).
    """

    name = "cosine"

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def _compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xn = x / x.norm(dim=1, keepdim=True).clamp_min(self.eps)
        yn = y / y.norm(dim=1, keepdim=True).clamp_min(self.eps)
        return xn @ yn.T

    def __repr__(self) -> str:  # pragma: no cover
        return f"CosineKernel(eps={self.eps})"

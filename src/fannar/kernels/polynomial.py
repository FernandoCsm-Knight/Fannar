"""Kernel polinomial: ``k(x, y) = (gamma * <x, y> + coef0)^degree``."""

from __future__ import annotations

import torch

from .base import BaseKernel


class PolynomialKernel(BaseKernel):
    """Kernel polinomial. PSD para ``coef0 >= 0`` e ``degree`` inteiro positivo."""

    name = "polynomial"

    def __init__(
        self, degree: int = 3, gamma: float = 1.0, coef0: float = 1.0
    ) -> None:
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    def _compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        base = self.gamma * (x @ y.T) + self.coef0
        return base ** self.degree

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PolynomialKernel(degree={self.degree}, gamma={self.gamma}, "
            f"coef0={self.coef0})"
        )

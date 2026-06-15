"""Kernel RBF (gaussiano): ``k(x, y) = exp(-gamma * ||x - y||^2)``."""

from __future__ import annotations

import torch

from ..utils.linalg import squared_distances
from .base import BaseKernel


class RBFKernel(BaseKernel):
    """Kernel gaussiano. Se ``gamma is None``, usa a heurística da mediana."""

    name = "rbf"

    def __init__(self, gamma: float | None = None) -> None:
        self.gamma = gamma

    def _resolve_gamma(self, d2: torch.Tensor) -> float:
        if self.gamma is not None:
            return float(self.gamma)
        # heurística da mediana sobre distâncias quadradas não nulas
        off = d2[~torch.eye(d2.shape[0], dtype=torch.bool, device=d2.device)] \
            if d2.shape[0] == d2.shape[1] else d2.flatten()
        med = off[off > 0].median() if (off > 0).any() else d2.new_tensor(1.0)
        med = med.clamp_min(1e-12)
        return float(1.0 / (2.0 * med))

    def _compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d2 = squared_distances(x, y)
        gamma = self._resolve_gamma(d2)
        return torch.exp(-gamma * d2)

    def __repr__(self) -> str:  # pragma: no cover
        return f"RBFKernel(gamma={self.gamma})"

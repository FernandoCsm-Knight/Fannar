"""Kernel linear: ``k(x, y) = <x, y>``."""

from __future__ import annotations

import torch

from .base import BaseKernel


class LinearKernel(BaseKernel):
    """Produto interno euclidiano. Sempre PSD na forma simétrica."""

    name = "linear"

    def _compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x @ y.T

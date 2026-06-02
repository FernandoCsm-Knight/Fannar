"""Normalização angular: ``K_ij / sqrt(K_ii K_jj)`` (kernel cosseno induzido)."""

from __future__ import annotations

import torch

from .base import BaseTransform


class AngularNormalizeTransform(BaseTransform):
    """Normaliza pela magnitude individual: produz produto interno cosseno.

    Diagonal resultante = 1 (para entradas com ``K_ii > 0``). Preserva PSD
    (congruência por diagonal positiva ``D^{-1/2} K D^{-1/2}``).
    """

    name = "angular_normalize"

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def _apply(self, K: torch.Tensor) -> torch.Tensor:
        d = torch.diagonal(K).clamp_min(self.eps).sqrt()  # (n,)
        return K / (d.unsqueeze(0) * d.unsqueeze(1))

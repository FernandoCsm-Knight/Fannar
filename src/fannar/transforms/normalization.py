"""Normalizações escalares de matrizes de Gram."""

from __future__ import annotations

import torch

from ..utils.linalg import frobenius_norm, top_eigenvalue, trace
from .base import BaseTransform


class TraceNormalizeTransform(BaseTransform):
    """``K / tr(K)``. Resultante tem traço 1 (se ``tr(K) > 0``). Preserva PSD."""

    name = "trace_normalize"

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def _apply(self, K: torch.Tensor) -> torch.Tensor:
        return K / trace(K).clamp_min(self.eps)


class FrobeniusNormalizeTransform(BaseTransform):
    """``K / ||K||_F``. Preserva PSD."""

    name = "frobenius_normalize"

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def _apply(self, K: torch.Tensor) -> torch.Tensor:
        return K / frobenius_norm(K).clamp_min(self.eps)


class MaxEigenvalueNormalizeTransform(BaseTransform):
    """``K / λ_max(K)``. Remove a escala do modo principal. Preserva PSD."""

    name = "max_eigenvalue_normalize"

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def _apply(self, K: torch.Tensor) -> torch.Tensor:
        return K / top_eigenvalue(K).clamp_min(self.eps)

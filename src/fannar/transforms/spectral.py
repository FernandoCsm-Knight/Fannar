"""Transformação espectral genérica ``T(K) = f(K)`` sobre o espectro.

Aplica uma função escalar aos autovalores. Quando ``f`` tem coeficientes não
negativos (ou é monótona não negativa sobre [0, λ_max]), preserva PSD.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from ..utils.linalg import eigh_descending, symmetrize
from .base import BaseTransform


class SpectralFunctionTransform(BaseTransform):
    """Aplica ``f`` aos autovalores e reconstrói. ``f`` opera elemento a elemento."""

    name = "spectral_function"

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor], clamp: bool = True) -> None:
        self.fn = fn
        self.clamp = clamp

    def _apply(self, K: torch.Tensor) -> torch.Tensor:
        dec = eigh_descending(K)
        lam = dec.eigenvalues
        if self.clamp:
            lam = lam.clamp_min(0.0)
        new = self.fn(lam)
        V = dec.eigenvectors
        return symmetrize((V * new.unsqueeze(0)) @ V.T)

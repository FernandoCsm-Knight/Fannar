"""Whitening espectral.

Decisão de modelagem: ``SpectralWhiteningTransform`` calcula a potência
espectral ``V diag((λ+eps)^power) Vᵀ``. Com ``power=-0.5`` obtém-se ``K^{-1/2}``.
Para ACHATAR o espectro (todas as direções com peso uniforme) use
``flatten=True``, que mapeia autovalores positivos para 1. Os dois usos são
distintos; a documentação da teoria (T_w) descreve o achatamento espectral.
"""

from __future__ import annotations

import torch

from ..utils.linalg import eigh_descending, symmetrize
from .base import BaseTransform


class SpectralWhiteningTransform(BaseTransform):
    """Whitening via potência espectral.

    Parameters
    ----------
    eps:
        Regularização somada aos autovalores antes de elevar à potência.
    power:
        Expoente aplicado aos autovalores (``-0.5`` => ``K^{-1/2}``).
    flatten:
        Se ``True``, ignora ``power`` e mapeia todo autovalor ``> eps`` para 1
        (espectro achatado, T_w da teoria).
    """

    name = "spectral_whitening"

    def __init__(self, eps: float = 1e-6, power: float = -0.5, flatten: bool = False) -> None:
        self.eps = eps
        self.power = power
        self.flatten = flatten

    def _apply(self, K: torch.Tensor) -> torch.Tensor:
        dec = eigh_descending(K)
        lam = dec.eigenvalues.clamp_min(0.0)
        if self.flatten:
            new = torch.where(lam > self.eps, torch.ones_like(lam), torch.zeros_like(lam))
        else:
            new = (lam + self.eps) ** self.power
        V = dec.eigenvectors
        return symmetrize((V * new.unsqueeze(0)) @ V.T)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SpectralWhiteningTransform(eps={self.eps}, power={self.power}, "
            f"flatten={self.flatten})"
        )

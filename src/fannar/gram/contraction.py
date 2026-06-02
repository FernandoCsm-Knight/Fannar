"""Contração espectral: reduz a Gram a coordenadas de baixa dimensão."""

from __future__ import annotations

from typing import Literal

import torch

from .eigenspace import eigendecompose

Weighting = Literal["sqrt_eigenvalue", "eigenvalue", "none"]


def spectral_contract(
    K: torch.Tensor, dim: int = 2, weighting: Weighting = "sqrt_eigenvalue"
) -> torch.Tensor:
    """Projeta os objetos em ``dim`` coordenadas espectrais.

    - ``sqrt_eigenvalue``: ``sqrt(λ_k) V_ik`` (MDS clássico, default).
    - ``eigenvalue``: ``λ_k V_ik``.
    - ``none``: ``V_ik`` (autovetores puros).
    """
    dec = eigendecompose(K, descending=True)
    lam = dec.eigenvalues.clamp_min(0.0)[:dim]
    V = dec.eigenvectors[:, :dim]
    if weighting == "sqrt_eigenvalue":
        return V * lam.sqrt().unsqueeze(0)
    if weighting == "eigenvalue":
        return V * lam.unsqueeze(0)
    if weighting == "none":
        return V
    raise ValueError(f"weighting '{weighting}' inválido.")

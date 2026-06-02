"""Grafo de afinidades e laplaciano normalizado.

A_ij = exp(-D_ij² / σ²);  L = I - Deg^{-1/2} A Deg^{-1/2}.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..utils.linalg import symmetrize


@dataclass
class LaplacianResult:
    affinity: torch.Tensor          # A
    degree: torch.Tensor            # vetor de graus (n,)
    laplacian: torch.Tensor         # L normalizado
    diffusion: torch.Tensor         # P = Deg^{-1} A
    sigma: float
    eigenvalues: torch.Tensor       # espectro de L (ascendente)


def median_sigma(D: torch.Tensor) -> float:
    """Heurística da mediana das distâncias não nulas para ``σ``."""
    n = D.shape[0]
    if n <= 1:
        return 1.0
    off = D[~torch.eye(n, dtype=torch.bool, device=D.device)]
    pos = off[off > 0]
    med = pos.median() if pos.numel() > 0 else torch.tensor(1.0, device=D.device)
    return float(med.clamp_min(1e-12))


def build_laplacian(
    D: torch.Tensor, sigma: float | None = None, eps: float = 1e-12
) -> LaplacianResult:
    """Constrói afinidade gaussiana, laplaciano normalizado e operador de difusão."""
    n = D.shape[0]
    if sigma is None:
        sigma = median_sigma(D)
    A = torch.exp(-(D ** 2) / (2.0 * sigma ** 2))
    A = symmetrize(A)
    deg = A.sum(dim=1)                                  # (n,)
    d_inv_sqrt = deg.clamp_min(eps).rsqrt()
    L = torch.eye(n, dtype=A.dtype, device=A.device) - (
        d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)
    )
    L = symmetrize(L)
    P = A / deg.clamp_min(eps).unsqueeze(1)             # Deg^{-1} A (linha-estocástico)
    evals = torch.linalg.eigvalsh(L)                    # ascendente
    return LaplacianResult(
        affinity=A, degree=deg, laplacian=L, diffusion=P, sigma=sigma, eigenvalues=evals
    )

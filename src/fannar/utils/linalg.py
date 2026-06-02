"""Rotinas de álgebra linear numérica usadas em toda a fannar.

Centraliza simetrização, decomposição espectral estável, potências espectrais
e projeção para a PSD mais próxima, com tolerâncias configuráveis.
"""

from __future__ import annotations

import torch

from ..config import get_config
from ..types import SpectralDecomposition


def symmetrize(K: torch.Tensor) -> torch.Tensor:
    """Retorna ``(K + Kᵀ) / 2``."""
    return 0.5 * (K + K.transpose(-1, -2))


def eigh_descending(K: torch.Tensor) -> SpectralDecomposition:
    """Decomposição espectral simétrica, autovalores em ordem decrescente.

    Usa ``torch.linalg.eigh`` (assume simetria). Caso a entrada não seja
    exatamente simétrica, ela é simetrizada antes.
    """
    Ks = symmetrize(K)
    evals, evecs = torch.linalg.eigh(Ks)  # ascending
    idx = torch.argsort(evals, descending=True)
    return SpectralDecomposition(
        eigenvalues=evals[idx].contiguous(),
        eigenvectors=evecs[:, idx].contiguous(),
        descending=True,
    )


def clamp_eigenvalues(evals: torch.Tensor) -> torch.Tensor:
    """Clampa autovalores negativos a zero (ruído numérico de PSD)."""
    return evals.clamp_min(0.0)


def spectral_power(
    K: torch.Tensor, power: float, eps: float | None = None, clamp: bool = True
) -> torch.Tensor:
    """Calcula ``f(K) = V diag((λ+eps)^power) Vᵀ`` (potência espectral).

    Para ``power`` negativo (ex.: whitening com -0.5), ``eps`` regulariza
    autovalores próximos de zero. Mantém simetria por construção.
    """
    eps = get_config().eps if eps is None else eps
    dec = eigh_descending(K)
    lam = dec.eigenvalues
    if clamp:
        lam = lam.clamp_min(0.0)
    powered = (lam + eps) ** power
    V = dec.eigenvectors
    return (V * powered.unsqueeze(0)) @ V.T


def nearest_psd(K: torch.Tensor, eps: float | None = None) -> torch.Tensor:
    """Projeta ``K`` na PSD simétrica mais próxima (em norma de Frobenius).

    Simetriza, zera autovalores negativos e reconstrói. Esta é a projeção
    exata de Higham para o cone PSD partindo da parte simétrica.
    """
    eps = get_config().eps_psd if eps is None else eps
    dec = eigh_descending(K)
    lam = dec.eigenvalues.clamp_min(0.0)
    V = dec.eigenvectors
    return symmetrize((V * lam.unsqueeze(0)) @ V.T)


def safe_divide(
    a: torch.Tensor, b: torch.Tensor | float, eps: float | None = None
) -> torch.Tensor:
    """Divisão protegida contra denominador ~0 (preserva sinal do denominador)."""
    eps = get_config().eps if eps is None else eps
    if isinstance(b, (int, float)):
        denom = b if abs(b) > eps else (eps if b >= 0 else -eps)
        return a / denom
    sign = torch.where(b >= 0, 1.0, -1.0)
    safe_b = torch.where(b.abs() > eps, b, sign * eps)
    return a / safe_b


def trace(K: torch.Tensor) -> torch.Tensor:
    return torch.diagonal(K, dim1=-2, dim2=-1).sum(-1)


def frobenius_norm(K: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_norm(K, ord="fro")


def top_eigenvalue(K: torch.Tensor) -> torch.Tensor:
    """Maior autovalor (em módulo simétrico) via eigvalsh."""
    return torch.linalg.eigvalsh(symmetrize(K)).max()

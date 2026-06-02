"""Operações relacionadas a semidefinição positiva (PSD)."""

from __future__ import annotations

import torch

from ..utils.linalg import nearest_psd as _nearest_psd
from ..utils.validation import is_psd

__all__ = ["is_psd", "project_psd", "clamp_negative_eigenvalues"]


def project_psd(K: torch.Tensor, eps: float | None = None) -> torch.Tensor:
    """Projeta ``K`` na PSD simétrica mais próxima (Frobenius)."""
    return _nearest_psd(K, eps=eps)


def clamp_negative_eigenvalues(K: torch.Tensor, eps: float | None = None) -> torch.Tensor:
    """Alias semântico de :func:`project_psd` (zera autovalores negativos)."""
    return _nearest_psd(K, eps=eps)

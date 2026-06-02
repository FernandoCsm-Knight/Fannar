"""Distância induzida pela norma do RKHS.

``d(i, j)^2 = K(i,i) + K(j,j) - 2 K(i,j)``  (Eq. de distância no RKHS).
"""

from __future__ import annotations

import torch

from ..utils.linalg import symmetrize


def distance_matrix(K: torch.Tensor, clamp: bool = True, squared: bool = False) -> torch.Tensor:
    """Matriz de distâncias ``D`` a partir da Gram ``K``.

    Parameters
    ----------
    clamp:
        Clampa distâncias quadradas negativas (ruído numérico) a zero.
    squared:
        Se ``True``, retorna ``D^2``; caso contrário ``D``.
    """
    K = symmetrize(K)
    diag = torch.diagonal(K)
    d2 = diag.unsqueeze(0) + diag.unsqueeze(1) - 2.0 * K
    if clamp:
        d2 = d2.clamp_min(0.0)
    # Força diagonal exatamente zero
    d2 = d2 - torch.diag(torch.diagonal(d2))
    if squared:
        return d2
    return d2.sqrt()

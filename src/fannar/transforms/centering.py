"""Identidade e centramento (``H K H``)."""

from __future__ import annotations

import torch

from .base import BaseTransform


class IdentityTransform(BaseTransform):
    """Não altera ``K``. Preserva estrutura absoluta de similaridade."""

    name = "identity"
    symmetrize_input = False

    def _apply(self, K: torch.Tensor) -> torch.Tensor:
        return K


class CenteringTransform(BaseTransform):
    """Centramento duplo ``H K H`` com ``H = I - 11ᵀ/n``.

    Remove o centroide no espaço de representação. Preserva simetria e PSD
    (``H`` é projetor ortogonal, e ``H K H`` é congruência de PSD).
    """

    name = "centering"

    def _apply(self, K: torch.Tensor) -> torch.Tensor:
        mean_rows = K.mean(dim=0, keepdim=True)      # (1,n)
        mean_cols = K.mean(dim=1, keepdim=True)      # (n,1)
        mean_all = K.mean()
        return K - mean_rows - mean_cols + mean_all

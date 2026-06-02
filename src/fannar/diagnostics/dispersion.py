"""Dispersão estrutural: distância média entre objetos."""

from __future__ import annotations

import torch

from ..types import DiagnosticResult


def dispersion(D: torch.Tensor) -> DiagnosticResult:
    """``Disp = mean(D_ij)`` (média sobre todos os pares, inclusive diagonal).

    Segue a definição da teoria ``(1/C²) Σ_ij d(i,j)``.
    """
    value = D.mean()
    return DiagnosticResult(
        name="dispersion",
        value=value,
        detail={"n": int(D.shape[0]), "max": float(D.max()), "min_offdiag": float(
            D[~torch.eye(D.shape[0], dtype=torch.bool, device=D.device)].min()
        ) if D.shape[0] > 1 else 0.0},
    )

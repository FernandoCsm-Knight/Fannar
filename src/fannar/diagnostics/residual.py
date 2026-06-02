"""Energia residual: fração de ``u`` fora do subespaço principal."""

from __future__ import annotations

import torch

from ..gram.eigenspace import energy_decomposition, principal_subspace
from ..types import DiagnosticResult, EnergyKind, Metric


def residual_energy(
    u: torch.Tensor,
    K: torch.Tensor,
    tau: float = 0.9,
    energy: EnergyKind = "operator",
    metric: Metric = "euclidean",
) -> DiagnosticResult:
    """``Res(u; tau) = ||P_perp u||² / ||u||²`` ∈ [0, 1].

    Alto => a estrutura interna do núcleo não captura ``u`` no nível ``tau``.
    """
    sub = principal_subspace(K, tau=tau, energy=energy)
    dec = energy_decomposition(u, sub, metric=metric)
    ratio = dec.residual_ratio
    return DiagnosticResult(
        name="residual_energy",
        value=ratio if ratio.ndim == 0 else ratio.mean(),
        detail={
            "residual_ratio": ratio,
            "captured": dec.captured,
            "residual": dec.residual,
            "r": sub.r,
            "tau": tau,
        },
    )

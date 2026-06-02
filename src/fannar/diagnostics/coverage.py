"""Cobertura espectral: fração de modos necessária para o limiar de energia."""

from __future__ import annotations

import torch

from ..gram.eigenspace import principal_subspace
from ..types import DiagnosticResult, EnergyKind


def coverage(
    K: torch.Tensor, tau: float = 0.95, energy: EnergyKind = "operator"
) -> DiagnosticResult:
    """``Cov(tau) = r(tau) / C``.

    Baixo => poucos modos concentram a energia (subespaço efetivo de baixa
    dimensão). Alto => energia distribuída por muitos modos.
    """
    sub = principal_subspace(K, tau=tau, energy=energy)
    C = sub.n
    value = sub.r / C
    return DiagnosticResult(
        name="coverage",
        value=value,
        detail={
            "r": sub.r,
            "C": C,
            "tau": tau,
            "energy": energy,
            "captured_fraction": float(sub.captured_fraction()),
        },
    )

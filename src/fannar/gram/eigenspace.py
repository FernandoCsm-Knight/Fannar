"""Decomposição espectral, subespaço principal e energia residual.

Implementa a Subseção "Complemento Ortogonal e Subespaço Principal do Núcleo"
da teoria: energia do operador (Σλ²) e energia por traço (Σλ), construção de
``V_r`` e razão de energia ortogonal.
"""

from __future__ import annotations

import torch

from ..types import (
    EnergyDecomposition,
    EnergyKind,
    Metric,
    PrincipalSubspace,
    SpectralDecomposition,
)
from ..utils.linalg import eigh_descending


def eigendecompose(K: torch.Tensor, descending: bool = True) -> SpectralDecomposition:
    """Decomposição espectral simétrica (autovalores decrescentes por padrão)."""
    dec = eigh_descending(K)
    if not descending:
        idx = torch.argsort(dec.eigenvalues, descending=False)
        return SpectralDecomposition(
            dec.eigenvalues[idx], dec.eigenvectors[:, idx], descending=False
        )
    return dec


def _energy_weights(eigenvalues: torch.Tensor, energy: EnergyKind) -> torch.Tensor:
    lam = eigenvalues.clamp_min(0.0)
    return lam ** 2 if energy == "operator" else lam


def select_rank(eigenvalues: torch.Tensor, tau: float, energy: EnergyKind = "operator") -> int:
    """Menor ``r`` tal que a fração de energia acumulada ``>= tau``."""
    if not (0.0 < tau <= 1.0):
        raise ValueError(f"tau deve estar em (0, 1], recebido {tau}.")
    w = _energy_weights(eigenvalues, energy)
    total = w.sum()
    if total <= 0:
        return 1
    cum = torch.cumsum(w, dim=0) / total
    # primeiro índice onde cum >= tau (1-based)
    meets = (cum >= tau - 1e-12).nonzero()
    if meets.numel() == 0:
        return int(eigenvalues.shape[0])
    return int(meets[0].item()) + 1


def principal_subspace(
    K: torch.Tensor, tau: float = 0.95, energy: EnergyKind = "operator"
) -> PrincipalSubspace:
    """Constrói o subespaço principal ``V_r`` de ``K`` para o limiar ``tau``."""
    dec = eigendecompose(K, descending=True)
    r = select_rank(dec.eigenvalues, tau, energy)
    return PrincipalSubspace(
        eigenvalues=dec.eigenvalues,
        eigenvectors=dec.eigenvectors,
        r=r,
        tau=tau,
        energy=energy,
    )


def energy_decomposition(
    u: torch.Tensor,
    subspace: PrincipalSubspace,
    metric: Metric = "euclidean",
    eps: float = 1e-12,
) -> EnergyDecomposition:
    """Decompõe ``u`` em componentes paralela (``V_r``) e ortogonal (``V_r^perp``).

    Convenções de métrica
    ----------------------
    - ``"euclidean"``: ``u`` é um vetor em ``R^n`` (ex.: valor por canal). Os
      autovetores formam uma base ortonormal; a projeção é ``P = V_r V_rᵀ`` e
      ``residual_ratio = ||u - Pu||² / ||u||²``. Esta é a definição direta da
      razão de energia ortogonal e é o default seguro.
    - ``"rkhs"``: ``u`` é interpretado com coordenadas ``a_k = <u, v_k>`` cuja
      norma RKHS pondera por ``1/λ_k`` (modos de pequeno autovalor "custam"
      mais). DECISÃO DE MODELAGEM: esta convenção segue a base ortonormal
      ``{sqrt(λ_k) φ_k}`` do RKHS; documente-a ao reportar resultados.

    ``u`` pode ter shape ``(n,)`` ou ``(n, m)`` (m colunas independentes).
    """
    V = subspace.eigenvectors            # (n, n)
    r = subspace.r
    if u.ndim == 1:
        u_col = u.unsqueeze(1)           # (n,1)
        single = True
    else:
        u_col = u
        single = False

    coeffs = V.T @ u_col                  # (n, m) coords na base de autovetores

    if metric == "euclidean":
        w = torch.ones_like(subspace.eigenvalues)
    elif metric == "rkhs":
        w = 1.0 / subspace.eigenvalues.clamp_min(eps)
    else:
        raise ValueError(f"metric '{metric}' inválida.")

    wcol = w.unsqueeze(1)                  # (n,1)
    weighted_sq = wcol * coeffs ** 2       # (n, m)

    captured = weighted_sq[:r].sum(dim=0)  # (m,)
    residual = weighted_sq[r:].sum(dim=0)  # (m,)
    total = captured + residual
    ratio = residual / total.clamp_min(eps)

    # componentes reconstruídas no espaço ambiente (R^n)
    parallel = V[:, :r] @ coeffs[:r]       # (n, m)
    orthogonal = V[:, r:] @ coeffs[r:]     # (n, m)

    if single:
        captured = captured.squeeze(0)
        residual = residual.squeeze(0)
        total = total.squeeze(0)
        ratio = ratio.squeeze(0)
        parallel = parallel.squeeze(1)
        orthogonal = orthogonal.squeeze(1)

    return EnergyDecomposition(
        total=total,
        captured=captured,
        residual=residual,
        residual_ratio=ratio,
        parallel=parallel,
        orthogonal=orthogonal,
    )


def spectral_coordinates(K: torch.Tensor, dim: int = 2) -> torch.Tensor:
    """Coordenadas espectrais ``z_i = (sqrt(λ_1) V_i1, ..., sqrt(λ_dim) V_idim)``.

    Retorna ``(n, dim)``. Equivale a MDS clássico sobre a Gram.
    """
    dec = eigendecompose(K, descending=True)
    lam = dec.eigenvalues.clamp_min(0.0)[:dim]
    V = dec.eigenvectors[:, :dim]
    return V * lam.sqrt().unsqueeze(0)


def embedding_fidelity(K: torch.Tensor, dim: int = 2) -> float:
    """Fidelidade de embedding ``Σ_{k<=dim} λ_k / Σ_k λ_k`` (cobertura por traço).

    Distinta da cobertura por energia do operador; mede preservação da soma
    de distâncias na projeção em ``dim`` dimensões.
    """
    dec = eigendecompose(K, descending=True)
    lam = dec.eigenvalues.clamp_min(0.0)
    total = lam.sum().clamp_min(1e-12)
    return float(lam[:dim].sum() / total)

"""Similaridade inter-camada para o eixo de unificação (α_Un).

Camadas têm tamanhos C_ℓ possivelmente distintos, então a comparação é feita
sobre invariantes da geometria que não dependem de identificação de canais entre
camadas.

Dois níveis de implementação:
- ``layer_similarity``: aproximação rápida via similaridade cosseno de assinaturas
  espectrais da Gram (mantida para retrocompatibilidade).
- ``layer_sim_theory``: fórmula exata da teoria (main.tex, Eq. similarity):
  Sim(ℓ,ℓ') = exp(-|H_esp(ℓ)-H_esp(ℓ')|/η_H - W₁(F_ℓ,F_ℓ')/η_W), onde H_esp
  usa o espectro do laplaciano normalizado e W₁ é a distância de Wasserstein-1
  entre distribuições empíricas das distâncias.

Funcionais espectrais do laplaciano (main.tex, subseção 4.2):
- ``spectral_entropy``: H_esp(ℓ) = -Σ μ̃_k log μ̃_k
- ``spectral_gap``:    Gap(ℓ) = μ₂ - μ₁
- ``effective_dimensionality``: D_eff(ℓ) = exp(H_esp(ℓ))
"""

from __future__ import annotations

import math

import torch

from ..gram.eigenspace import eigendecompose

# ---------------------------------------------------------------------------
# Funcionais espectrais do laplaciano
# ---------------------------------------------------------------------------

def spectral_entropy(eigenvalues: torch.Tensor) -> float:
    """H_esp = -Σ μ̃_k log μ̃_k  (autovalores do laplaciano normalizados pelo total).

    ``eigenvalues`` deve ser o espectro do laplaciano normalizado (ascendente),
    conforme retornado por ``LaplacianResult.eigenvalues``.
    """
    mu = eigenvalues.clamp_min(0.0)
    total = mu.sum().clamp_min(1e-12)
    mu_n = (mu / total).clamp_min(1e-12)
    return float(-(mu_n * mu_n.log()).sum())


def spectral_gap(eigenvalues: torch.Tensor) -> float:
    """Gap(ℓ) = μ₂ - μ₁ do laplaciano normalizado (autovalores ascendentes)."""
    if eigenvalues.shape[0] < 2:
        return 0.0
    return float(eigenvalues[1] - eigenvalues[0])


def effective_dimensionality(eigenvalues: torch.Tensor) -> float:
    """D_eff(ℓ) = exp(H_esp(ℓ))."""
    return math.exp(spectral_entropy(eigenvalues))


# ---------------------------------------------------------------------------
# Similaridade inter-camada
# ---------------------------------------------------------------------------

def _spectral_signature(K: torch.Tensor, m: int = 16) -> torch.Tensor:
    """Assinatura: top-``m`` autovalores de K normalizados pelo traço (padded)."""
    lam = eigendecompose(K, descending=True).eigenvalues.clamp_min(0.0)
    total = lam.sum().clamp_min(1e-12)
    lam = lam / total
    sig = torch.zeros(m, dtype=lam.dtype, device=lam.device)
    k = min(m, lam.shape[0])
    sig[:k] = lam[:k]
    return sig


def layer_similarity(K_a: torch.Tensor, K_b: torch.Tensor, m: int = 16) -> float:
    """Similaridade cosseno entre assinaturas espectrais de duas camadas ∈ [0, 1].

    Aproximação rápida que não requer matriz de distâncias. Para a fórmula exata
    da teoria, use ``layer_sim_theory``.
    """
    sa = _spectral_signature(K_a, m)
    sb = _spectral_signature(K_b, m)
    num = (sa * sb).sum()
    den = sa.norm() * sb.norm()
    if den <= 1e-12:
        return 0.0
    return float((num / den).clamp(0.0, 1.0))


def layer_sim_theory(
    D_a: torch.Tensor,
    D_b: torch.Tensor,
    *,
    eta_H: float = 1.0,
    eta_W: float = 1.0,
    sigma_a: float | None = None,
    sigma_b: float | None = None,
) -> float:
    """Sim(ℓ, ℓ') conforme a fórmula da teoria (main.tex, Eq. similarity).

    Sim = exp(-|H_esp(ℓ) - H_esp(ℓ')| / η_H  -  W₁(F_ℓ, F_ℓ') / η_W)

    ``D_a`` e ``D_b`` são matrizes de distâncias (n×n) das duas camadas.
    ``eta_H`` e ``eta_W`` são as escalas de normalização; por convenção da
    teoria, calibram-se sobre a referência como a mediana das discrepâncias
    entre pares de camadas — passe os valores calibrados se disponíveis.

    Requer ``scipy`` para W₁. Se não disponível, a componente W₁ é omitida
    (termo zero) e um aviso é emitido; o resultado permanece em [0, 1] mas
    reflete apenas a divergência de entropia espectral.
    """
    import warnings

    from .laplacian import build_laplacian

    lap_a = build_laplacian(D_a, sigma=sigma_a)
    lap_b = build_laplacian(D_b, sigma=sigma_b)

    H_a = spectral_entropy(lap_a.eigenvalues)
    H_b = spectral_entropy(lap_b.eigenvalues)
    term_H = abs(H_a - H_b) / max(eta_H, 1e-12)

    # Wasserstein-1 entre distribuições empíricas de distâncias (pares off-diag)
    n_a, n_b = D_a.shape[0], D_b.shape[0]
    mask_a = ~torch.eye(n_a, dtype=torch.bool, device=D_a.device)
    mask_b = ~torch.eye(n_b, dtype=torch.bool, device=D_b.device)
    d_a_np = D_a[mask_a].cpu().numpy()
    d_b_np = D_b[mask_b].cpu().numpy()

    try:
        from scipy.stats import wasserstein_distance  # type: ignore[import]
        w1 = float(wasserstein_distance(d_a_np, d_b_np))
    except ImportError:
        warnings.warn(
            "scipy não encontrado: componente W₁ de layer_sim_theory omitida. "
            "Instale scipy para a fórmula completa.",
            stacklevel=2,
        )
        w1 = 0.0

    term_W = w1 / max(eta_W, 1e-12)
    return float(math.exp(-(term_H + term_W)))

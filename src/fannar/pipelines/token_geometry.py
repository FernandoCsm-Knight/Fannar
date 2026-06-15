"""Geometria token-a-token: distâncias RKHS entre tokens de uma sequência.

O tensor métrico combinado é M = K_W² + K_A² + K_G², onde cada K² é a
potência espectral da Gram correspondente (V Λ² Vᵀ), de modo que:

    d²(tokenᵢ, tokenⱼ) = d_W² + d_A² + d_G²
                        = (aᵢ−aⱼ)ᵀ M (aᵢ−aⱼ)

O acúmulo tensorial entre camadas reforça separações que aparecem de forma
consistente ao longo da hierarquia e atenua relações inconsistentes.
"""

from __future__ import annotations

import torch

from ..gram.distance import cumulative_tensor_distances, pairwise_sq_dists
from ..types import SpectralDecomposition


def _gram_squared(dec: SpectralDecomposition) -> torch.Tensor:
    """Retorna K² = V Λ² Vᵀ a partir da decomposição espectral de K = V Λ Vᵀ."""
    lam2 = dec.eigenvalues.clamp_min(0.0).pow(2)
    V = dec.eigenvectors
    return (V * lam2.unsqueeze(0)) @ V.T


def token_distance_matrix(
    activations: torch.Tensor,
    decomp_W: SpectralDecomposition,
    decomp_A: SpectralDecomposition | None = None,
    decomp_G: SpectralDecomposition | None = None,
) -> torch.Tensor:
    """Matriz de distâncias RKHS (S, S) entre os S tokens de uma camada.

    Para cada par de tokens (i, j) com diferença centrada ``Δ = aᵢ − aⱼ``:

    .. math::
        d^2(i, j) = \\Delta^\\top (K_W^2 + K_A^2 + K_G^2)\\, \\Delta

    onde ``K_X^2 = V_X \\Lambda_X^2 V_X^\\top`` é a potência espectral da
    Gram de cada fonte (pesos, ativações, gradientes).

    Parameters
    ----------
    activations : (S, C)  ativações centradas por token (``a -= a.mean(0)``).
    decomp_W    : decomposição espectral da Gram de pesos (C, C).
    decomp_A    : decomposição espectral da Gram de ativações (C, C) ou ``None``.
    decomp_G    : decomposição espectral da Gram de gradientes (C, C) ou ``None``.

    Returns
    -------
    D : (S, S) matriz simétrica de distâncias entre tokens, diagonal = 0.
    """
    M = _gram_squared(decomp_W)
    if decomp_A is not None:
        M = M + _gram_squared(decomp_A)
    if decomp_G is not None:
        M = M + _gram_squared(decomp_G)

    A = activations.float()
    M = M.to(device=A.device, dtype=A.dtype)
    d2 = pairwise_sq_dists(A, metric=M)
    D = d2.clamp_min(0.0).sqrt()
    D.fill_diagonal_(0.0)
    return D


def cumulative_token_distances(
    token_dist_by_layer: dict[int, torch.Tensor] | list[torch.Tensor],
) -> list[torch.Tensor]:
    """Distâncias acumuladas via produto tensorial de afinidades por camada.

    Wrapper semântico de :func:`~fannar.gram.distance.cumulative_tensor_distances`
    especializado para sequências de tokens.

    Cada passo acumula ``K_global = K_0 ⊙ ... ⊙ K_ℓ`` (produto de Hadamard
    das afinidades gaussianas) e converte de volta em distância.

    Parameters
    ----------
    token_dist_by_layer : ``dict {layer_idx: D (S,S)}`` ou lista de ``D``.

    Returns
    -------
    Lista de matrizes de distâncias acumuladas, uma por camada.
    """
    return cumulative_tensor_distances(token_dist_by_layer)

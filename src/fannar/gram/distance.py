"""Distâncias e estrutura de curvas de nível no RKHS.

Camadas de acesso à geometria induzida por K:

1. ``distance_matrix``             — todas as distâncias par a par d(i,j).
2. ``pairwise_sq_dists``           — distâncias quadradas entre vetores brutos (euclidiana ou Mahalanobis).
3. ``affinity_from_distance``      — converte D em afinidade gaussiana.
4. ``cumulative_tensor_distances`` — acúmulo tensorial de afinidades por camada.
5. ``sentence_distance_matrix``    — D via folheação centrada em cada ponto.
6. ``level_crossing_matrix``       — contagem de hipersuperfícies cruzadas.
7. ``distance_to_ref``             — função distância p centrada em x₀.
8. ``level_set``                   — aproximação discreta de Γ_r(x₀).
9. ``level_density``               — densidade direcional ρ(xᵢ; v).
"""

from __future__ import annotations

import torch

from ..utils.linalg import squared_distances as _euclidean_sq
from ..utils.linalg import symmetrize


# ---------------------------------------------------------------------------
# 0. Distâncias quadradas entre vetores brutos
# ---------------------------------------------------------------------------

def pairwise_sq_dists(A: torch.Tensor, metric: torch.Tensor | None = None) -> torch.Tensor:
    """Distâncias quadradas par a par (N, N) entre as linhas de ``A``.

    Parameters
    ----------
    A      : (N, d) matriz de N vetores de dimensão d.
    metric : (d, d) tensor de métrica PSD simétrico.
        Se fornecido: ``d²(i,j) = (Aᵢ−Aⱼ)ᵀ M (Aᵢ−Aⱼ)`` (Mahalanobis).
        Se ``None``: distância euclidiana padrão.

    Returns
    -------
    D2 : (N, N) distâncias quadradas, diagonal = 0.
    """
    if metric is None:
        return _euclidean_sq(A, A)
    AMAT = A @ metric @ A.T
    diag = AMAT.diagonal()
    return (diag.unsqueeze(1) - 2.0 * AMAT + diag.unsqueeze(0)).clamp_min(0.0)


# ---------------------------------------------------------------------------
# 1. Distância par a par a partir da Gram
# ---------------------------------------------------------------------------

def distance_matrix(K: torch.Tensor, clamp: bool = True, squared: bool = False) -> torch.Tensor:
    """Matriz de distâncias ``D`` a partir da Gram ``K``.

    ``d(i,j)² = K(i,i) + K(j,j) - 2K(i,j)``

    Parameters
    ----------
    clamp:
        Clampa distâncias quadradas negativas (ruído numérico) a zero.
    squared:
        Se ``True``, retorna ``D²``; caso contrário ``D``.
    """
    K = symmetrize(K)
    diag = torch.diagonal(K)
    d2 = diag.unsqueeze(0) + diag.unsqueeze(1) - 2.0 * K
    if clamp:
        d2 = d2.clamp_min(0.0)
    d2 = d2 - torch.diag(torch.diagonal(d2))
    if squared:
        return d2
    return d2.sqrt()


# ---------------------------------------------------------------------------
# 1b. Afinidade gaussiana e acúmulo tensorial
# ---------------------------------------------------------------------------

def affinity_from_distance(D: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """Converte uma matriz de distâncias em afinidade gaussiana.

    .. math::
        K_{ij} = \\exp\\!\\left(-\\frac{D_{ij}^2}{2\\sigma^2}\\right)

    com ``σ`` igual à mediana das distâncias fora da diagonal (heurística)
    se não fornecido.  A diagonal é forçada a 1.

    Parameters
    ----------
    D     : (N, N) matriz de distâncias simétrica.
    sigma : desvio padrão da gaussiana; se ``None``, usa a mediana de D_off.

    Returns
    -------
    K : (N, N) afinidade gaussiana PSD.
    """
    mask = ~torch.eye(D.shape[0], dtype=torch.bool, device=D.device)
    if sigma is None:
        sigma = float(D[mask].median().clamp_min(1e-12))
    K = torch.exp(-D.pow(2) / (2.0 * max(sigma, 1e-12) ** 2))
    K.fill_diagonal_(1.0)
    return K


def cumulative_tensor_distances(
    dist_by_layer: dict[int, torch.Tensor] | list[torch.Tensor],
) -> list[torch.Tensor]:
    """Distâncias acumuladas via produto tensorial de afinidades por camada.

    Para cada camada ℓ:

    1. Converte ``D_ℓ`` em afinidade ``K_ℓ = affinity_from_distance(D_ℓ)``.
    2. Acumula ``K_global = K_0 ⊙ K_1 ⊙ ... ⊙ K_ℓ`` (produto de Hadamard).
    3. Converte de volta: ``D_cumul_ℓ = sentence_distance_matrix(K_global)``.

    O produto de Hadamard reforça separações consistentes ao longo da hierarquia:
    pares com afinidade alta em todas as camadas mantêm afinidade alta no global;
    pares com afinidade baixa em qualquer camada perdem afinidade.

    Parameters
    ----------
    dist_by_layer : ``dict {layer_idx: D (S,S)}`` ou lista de ``D (S,S)``.

    Returns
    -------
    Lista de matrizes de distâncias acumuladas, uma por camada (mesma ordem).
    """
    if isinstance(dist_by_layer, dict):
        layers = sorted(dist_by_layer.keys())
        matrices = [dist_by_layer[li] for li in layers]
    else:
        matrices = list(dist_by_layer)

    K_global: torch.Tensor | None = None
    cumulative: list[torch.Tensor] = []
    for D in matrices:
        K_layer = affinity_from_distance(D)
        K_global = K_layer if K_global is None else K_global * K_layer
        cumulative.append(sentence_distance_matrix(K_global))

    return cumulative


# ---------------------------------------------------------------------------
# 1c. Matriz de distâncias via folheação centrada em cada ponto
# ---------------------------------------------------------------------------

def sentence_distance_matrix(K: torch.Tensor) -> torch.Tensor:
    """Matriz de distâncias (N, N) construída via ``distance_to_ref``.

    Para cada ponto ``xᵢ`` como referência, calcula a função distância
    ``p`` centrada em ``xᵢ`` e monta a coluna correspondente da matriz ``D``.
    Isso expõe a estrutura da folheação: cada coluna é o campo escalar das
    distâncias a um ponto fixo, não uma fórmula all-pairs direta.

    Parameters
    ----------
    K : (N, N) matriz de Gram entre as N amostras.

    Returns
    -------
    D : (N, N) matriz simétrica de distâncias RKHS.
    """
    n = K.shape[0]
    return torch.stack([distance_to_ref(K, i) for i in range(n)], dim=1)


# ---------------------------------------------------------------------------
# 1c. Matriz de cruzamentos de nível
# ---------------------------------------------------------------------------

def level_crossing_matrix(
    D: torch.Tensor,
    n_levels: int = 12,
    quantile: float = 0.90,
) -> torch.Tensor:
    """Matriz de cruzamentos de nível: quantas hipersuperfícies Γ_r separam i de j.

    Define um espaçamento uniforme ``δ = q(D_off, p) / n_levels``, onde
    ``q(D_off, p)`` é o quantil ``p`` das distâncias fora da diagonal. Então

    .. math::
        C[i,j] = \\lceil D[i,j] / \\delta \\rceil

    corresponde ao número líquido de níveis ``\\{\\Gamma_{k\\delta}\\}_{k=1}^{n}``
    cruzados entre ``x_i`` e ``x_j``, conforme a Proposição 2 da teoria
    (``n(\\gamma, \\Delta r) \\cdot \\Delta r \\to p(x_1)`` quando ``\\Delta r \\to 0``).

    Parameters
    ----------
    D         : (N, N) matriz de distâncias (saída de ``sentence_distance_matrix``).
    n_levels  : número de níveis uniformes usados para discretizar.
    quantile  : quantil das distâncias fora da diagonal usado para calibrar δ.

    Returns
    -------
    C : (N, N) tensor inteiro de contagens de cruzamentos (diagonal = 0).
    """
    mask = ~torch.eye(D.shape[0], dtype=torch.bool, device=D.device)
    delta = torch.quantile(D[mask], quantile) / n_levels
    delta = delta.clamp_min(torch.finfo(D.dtype).tiny)
    C = torch.ceil(D / delta)
    C.fill_diagonal_(0.0)
    return C


# ---------------------------------------------------------------------------
# 2. Função distância centrada em x₀
# ---------------------------------------------------------------------------

def distance_to_ref(K: torch.Tensor, ref_idx: int) -> torch.Tensor:
    """Função distância ``p`` centrada em ``x_ref``.

    Para cada ponto ``xᵢ`` do conjunto de dados retorna

    .. math::
        p(x_i) = \\|\\Phi(x_i) - \\Phi(x_{\\mathrm{ref}})\\|_{\\mathcal{H}_K}
                = \\sqrt{K_{ii} + K_{\\mathrm{rr}} - 2K_{i\\mathrm{r}}}

    onde ``r = ref_idx``.  O valor em ``i = ref_idx`` é exatamente zero.

    Parameters
    ----------
    K       : (n, n) matriz de Gram simétrica PSD.
    ref_idx : índice do ponto de referência ``x₀``.

    Returns
    -------
    p : (n,) tensor de distâncias ao ponto de referência.
    """
    K = symmetrize(K)
    diag = K.diagonal()
    k_rr = K[ref_idx, ref_idx]
    k_ir = K[:, ref_idx]
    d2 = (diag + k_rr - 2.0 * k_ir).clamp_min(0.0)
    d2[ref_idx] = 0.0
    return d2.sqrt()


# ---------------------------------------------------------------------------
# 3. Curvas / hipersuperfícies de nível
# ---------------------------------------------------------------------------

def level_set(
    p: torch.Tensor,
    r: float,
    tol: float | None = None,
) -> torch.Tensor:
    """Aproximação discreta da hipersuperfície de nível Γ_r(x₀).

    Retorna os índices ``i`` tais que ``|p(xᵢ) - r| ≤ tol``, isto é, os
    pontos equidistantes de ``x₀`` na métrica do RKHS com tolerância ``tol``.

    Parameters
    ----------
    p   : (n,) vetor de distâncias ao ponto de referência (saída de
          ``distance_to_ref``).
    r   : raio da hipersuperfície de nível.
    tol : semi-largura da faixa aceita.  Se ``None``, usa metade da mediana
          dos espaçamentos entre valores distintos de ``p`` — adaptativo ao
          espalhamento real do conjunto de dados.

    Returns
    -------
    indices : (m,) tensor com os índices que pertencem a Γ_r(x₀).
    """
    if tol is None:
        sorted_p, _ = p.sort()
        gaps = sorted_p[1:] - sorted_p[:-1]
        positive = gaps[gaps > 0]
        tol = float(positive.median() / 2.0) if positive.numel() > 0 else 1e-3

    return torch.where(torch.abs(p - r) <= tol)[0]


# ---------------------------------------------------------------------------
# 4. Densidade direcional de curvas de nível
# ---------------------------------------------------------------------------

def level_density(
    K: torch.Tensor,
    ref_idx: int,
    v: torch.Tensor,
    eval_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Densidade direcional ``ρ(xᵢ; v)`` da folheação centrada em ``x_ref``.

    Para um ponto ``xᵢ ≠ x₀``, a densidade direcional da função ``p`` na
    direção RKHS ``v = Σⱼ αⱼ K(·, xⱼ)`` (representada pelos coeficientes
    ``v ∈ ℝⁿ``) é

    .. math::
        \\rho(x_i; v)
        = \\bigl|\\langle \\nabla p(x_i),\\, v \\rangle_{\\mathcal{H}_K}\\bigr|
        = \\frac{\\bigl|(K_{i,:} - K_{\\mathrm{ref},:})\\, v\\bigr|}{p(x_i)}

    onde o gradiente de ``p`` em ``xᵢ`` no RKHS é
    ``∇p(xᵢ) = (Φ(xᵢ) - Φ(x₀)) / p(xᵢ)``.

    Valores altos de ρ indicam regiões em que pequenos deslocamentos na
    direção ``v`` produzem grande variação na representação; valores baixos
    indicam estabilidade local.  A função não está definida em ``xᵢ = x₀``
    (``p = 0``).

    Parameters
    ----------
    K            : (n, n) matriz de Gram simétrica PSD.
    ref_idx      : índice do ponto de referência ``x₀``.
    v            : (n,) coeficientes da direção RKHS
                   ``v = Σⱼ αⱼ Φ(xⱼ)``.
    eval_indices : (m,) índices em que avaliar ρ.  Se ``None``, usa todos
                   os pontos exceto ``ref_idx``.

    Returns
    -------
    rho : (m,) densidades direcionais nos pontos pedidos.
    """
    K = symmetrize(K)
    n = K.shape[0]

    if eval_indices is None:
        all_idx = torch.arange(n, device=K.device)
        eval_indices = all_idx[all_idx != ref_idx]

    p = distance_to_ref(K, ref_idx)

    # (Φ(xᵢ) - Φ(x₀)) em coordenadas da Gram: K[i,:] - K[ref,:]
    diff = K[eval_indices, :] - K[ref_idx, :]   # (m, n)
    numerator = torch.abs(diff @ v)              # (m,)
    denominator = p[eval_indices].clamp_min(torch.finfo(K.dtype).tiny)

    return numerator / denominator

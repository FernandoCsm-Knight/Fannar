"""Perfil explicativo: oito assinaturas geométricas (modelos de explicação).

Cada α_k ∈ [0,1] é uma fração genuína sobre objetos já construídos (espectro,
massa direcional, massa de difusão), conforme a Subseção "Perfil Explicativo
da Camada" da teoria. Eixos indefinidos retornam ``available=False`` em vez de
``NaN``.

Convenções importantes (decisões de modelagem explícitas):
- ``gradient`` é um vetor ``(C,)`` de relevância por canal (``g_i``). Se ``None``,
  os eixos que dependem dele (IS, RE, In) ficam indisponíveis.
- ``other_layer_grams`` é necessário para Un (requer >= 1 outra camada).
- Pr requer estatísticas inter-contexto (rótulos); passe ``pr_sep``/``pr_var``
  pré-computados, senão Pr fica indisponível.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..types import DiagnosticResult, EnergyKind, ExplanatoryProfile
from .coverage import coverage
from .inter_layer import layer_sim_theory, layer_similarity
from .laplacian import LaplacianResult, build_laplacian
from .residual import residual_energy
from .stability import stability


def _f(x: torch.Tensor | float | None) -> float:
    """Converte um valor de DiagnosticResult (Tensor|float|None) em float."""
    if x is None:
        return 0.0
    if isinstance(x, torch.Tensor):
        return float(x.item())
    return float(x)


def _normalized_entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    p = p / p.sum()
    H = -(p * p.log()).sum()
    C = p.shape[0]
    if C <= 1:
        return torch.tensor(0.0, dtype=p.dtype, device=p.device)
    return H / torch.log(torch.tensor(float(C), dtype=p.dtype, device=p.device))


def alpha_DN(K: torch.Tensor, lap: LaplacianResult, tau_dn: float = 0.95) -> DiagnosticResult:
    cov = coverage(K, tau=tau_dn, energy="operator")
    mu = lap.eigenvalues  # ascendente
    gap = (mu[1] - mu[0]) if mu.shape[0] >= 2 else torch.tensor(0.0, device=mu.device)
    val = (1.0 - _f(cov.value)) * float(gap) / 2.0
    return DiagnosticResult(
        "DN", max(0.0, min(1.0, val)), detail={"gap": float(gap), "cov": _f(cov.value)}
    )


def alpha_IS(K: torch.Tensor, gradient: torch.Tensor | None, tau_is: float = 0.9) -> DiagnosticResult:
    if gradient is None:
        return DiagnosticResult("IS", None, available=False)
    res = residual_energy(gradient, K, tau=tau_is, metric="euclidean")
    val = 1.0 - _f(res.value)
    return DiagnosticResult("IS", max(0.0, min(1.0, val)), detail={"residual": _f(res.value)})


def alpha_RE(gradient: torch.Tensor | None) -> DiagnosticResult:
    if gradient is None:
        return DiagnosticResult("RE", None, available=False)
    p = gradient ** 2
    if p.sum() <= 0:
        return DiagnosticResult("RE", 0.0, detail={"degenerate": True})
    Hn = _normalized_entropy(p)
    val = 1.0 - float(Hn)
    return DiagnosticResult("RE", max(0.0, min(1.0, val)), detail={"entropy_norm": float(Hn)})


def alpha_Pr(
    pr_sep: float | None = None, pr_var: float | None = None, eps: float = 1e-12
) -> DiagnosticResult:
    if pr_sep is None or pr_var is None:
        return DiagnosticResult("Pr", None, available=False,
                                detail={"reason": "requer estatísticas inter-contexto (rótulos)"})
    val = pr_sep / (pr_sep + pr_var + eps)
    return DiagnosticResult("Pr", max(0.0, min(1.0, val)))


def alpha_Un(
    K: torch.Tensor,
    other_layer_grams: Sequence[torch.Tensor] | None,
    D: torch.Tensor | None = None,
    other_layer_distances: Sequence[torch.Tensor] | None = None,
    eta_H: float = 1.0,
    eta_W: float = 1.0,
    m: int = 16,
) -> DiagnosticResult:
    """α_Un: fração média de Sim(ℓ, ℓ') sobre as demais camadas.

    Se ``D`` e ``other_layer_distances`` forem fornecidos, usa a fórmula exata
    da teoria (Eq. similarity via entropia espectral do laplaciano + Wasserstein).
    Caso contrário, usa a aproximação rápida por similaridade cosseno de assinaturas
    espectrais da Gram (``layer_similarity``).
    """
    if not other_layer_grams and not other_layer_distances:
        return DiagnosticResult("Un", None, available=False,
                                detail={"reason": "requer >= 1 outra camada"})

    use_theory = D is not None and other_layer_distances is not None and len(other_layer_distances) > 0

    if use_theory:
        sims = [layer_sim_theory(D, Dp, eta_H=eta_H, eta_W=eta_W)
                for Dp in other_layer_distances]  # type: ignore[union-attr]
    elif other_layer_grams:
        sims = [layer_similarity(K, Kp, m=m) for Kp in other_layer_grams]
    else:
        return DiagnosticResult("Un", None, available=False,
                                detail={"reason": "requer >= 1 outra camada"})

    val = float(sum(sims) / len(sims))
    method = "theory" if use_theory else "cosine_approx"
    return DiagnosticResult("Un", max(0.0, min(1.0, val)),
                            detail={"per_layer": sims, "method": method})


def alpha_CM(lap: LaplacianResult, t: int = 2) -> DiagnosticResult:
    P = lap.diffusion
    Pt = torch.linalg.matrix_power(P, t)
    val = float(torch.diagonal(Pt).mean())
    return DiagnosticResult("CM", max(0.0, min(1.0, val)), detail={"t": t})


def alpha_NM(lap: LaplacianResult, eps_cluster: float = 0.1) -> DiagnosticResult:
    mu = lap.eigenvalues  # ascendente
    C = mu.shape[0]
    k_star = int((mu < eps_cluster).sum().item())
    if not (2 <= k_star <= C // 2):
        return DiagnosticResult("NM", 0.0, detail={"k_star": k_star, "gated": True})
    gap = float(mu[k_star]) if k_star < C else 0.0  # μ_{k*+1} (0-indexed => mu[k_star])
    val = gap / 2.0
    return DiagnosticResult("NM", max(0.0, min(1.0, val)), detail={"k_star": k_star, "gap": gap})


def alpha_In(
    gradient: torch.Tensor | None, D: torch.Tensor, k_stab: int = 5
) -> DiagnosticResult:
    """α_In: maior combinação de relevância relativa e isolamento por canal.

    Coerência com α_RE
    ------------------
    r_i = g_i² / max_j(g_j²) tem max(r) = 1 e ι_i = Stab(i)/max(Stab) tem
    max(ι) = 1 por construção — logo max(r·ι) pode atingir 1 mesmo quando o
    gradiente é uniforme (RE ≈ 0), tornando In um falso positivo. Para garantir
    que In = 0 quando nenhum canal se destaca em relevância, o escore bruto é
    ponderado por α_RE: se o gradiente é uniforme, In colapsa a 0 independente
    do isolamento estrutural.
    """
    if gradient is None:
        return DiagnosticResult("In", None, available=False)
    stab = stability(D, k=k_stab)
    per_item = stab.detail["per_item"]
    g2 = gradient ** 2
    if g2.sum() <= 0:
        return DiagnosticResult("In", 0.0, detail={"degenerate": True})
    gmax = g2.max().clamp_min(1e-12)
    r = g2 / gmax
    smax = per_item.max().clamp_min(1e-12)
    iota = per_item / smax
    score = float((r * iota).max())          # escore bruto ∈ [0, 1]
    re_weight = float(1.0 - _normalized_entropy(g2))  # = α_RE
    val = score * re_weight
    return DiagnosticResult(
        "In", max(0.0, min(1.0, val)),
        detail={"score_raw": score, "re_weight": re_weight},
    )


def explanatory_profile(
    K: torch.Tensor,
    D: torch.Tensor | None = None,
    *,
    gradient: torch.Tensor | None = None,
    other_layer_grams: Sequence[torch.Tensor] | None = None,
    other_layer_distances: Sequence[torch.Tensor] | None = None,
    pr_sep: float | None = None,
    pr_var: float | None = None,
    tau_dn: float = 0.95,
    tau_is: float = 0.9,
    eps_cluster: float = 0.1,
    diffusion_t: int = 2,
    k_stab: int = 5,
    sigma: float | None = None,
    eta_H: float = 1.0,
    eta_W: float = 1.0,
    energy: EnergyKind = "operator",  # noqa: ARG001 - reservado p/ extensão
) -> ExplanatoryProfile:
    """Computa o perfil explicativo de 8 eixos para uma camada.

    Parâmetros
    ----------
    other_layer_distances:
        Matrizes de distâncias das demais camadas. Se fornecido junto com ``D``,
        α_Un usa a fórmula exata da teoria (Sim via entropia + Wasserstein).
        Caso contrário, ``other_layer_grams`` é usado com aproximação cosseno.
    eta_H, eta_W:
        Escalas de normalização para a fórmula Sim da teoria. Calibrar sobre
        a referência como mediana das discrepâncias entre pares de camadas.
    """
    from ..gram.distance import distance_matrix

    if D is None:
        D = distance_matrix(K)
    lap = build_laplacian(D, sigma=sigma)

    return ExplanatoryProfile(
        DN=alpha_DN(K, lap, tau_dn=tau_dn),
        IS=alpha_IS(K, gradient, tau_is=tau_is),
        RE=alpha_RE(gradient),
        Pr=alpha_Pr(pr_sep, pr_var),
        Un=alpha_Un(K, other_layer_grams, D=D,
                    other_layer_distances=other_layer_distances,
                    eta_H=eta_H, eta_W=eta_W),
        CM=alpha_CM(lap, t=diffusion_t),
        NM=alpha_NM(lap, eps_cluster=eps_cluster),
        In=alpha_In(gradient, D, k_stab=k_stab),
    )

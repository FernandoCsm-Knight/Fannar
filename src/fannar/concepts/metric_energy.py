from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class OrthogonalEnergySurface:
    """Superfície de energia no complemento ortogonal V_r⊥ do subespaço principal.

    Campos
    ------
    perp_energy : (H, W)
        Energia absoluta da projeção de cada localização espacial em V_r⊥.
        Alto => ativação nessa região não é capturada pelos modos principais.
    perp_ratio : (H, W)
        Fração da energia total em V_r⊥ — análogo espacial de E_ratio(u).
        Valores próximos de 1 indicam 'ponto cego' do núcleo nessa localização.
    total_energy : (H, W)
        Energia total (||x_s||²) por localização, para referência.
    grid : (H, W)
        Dimensões espaciais da grade de ativação.
    r : int
        Número de modos retidos em V_r (determinado por tau).
    tau : float
        Limiar de energia usado para selecionar V_r.
    coverage : float
        Fração de energia de K efetivamente retida em V_r (≥ tau por construção).
    """

    perp_energy: torch.Tensor
    perp_ratio: torch.Tensor
    total_energy: torch.Tensor
    grid: tuple[int, int]
    r: int
    tau: float
    coverage: float


@dataclass
class GramEnergySurface:
    """Superfície de energia induzida pela Gram no campo de ativações.

    Campos
    ------
    field : (H, W)
        Mapa suavizado da energia ponderada pelos autovalores da Gram.
        Alto => localização espacial tem alta projeção nos modos principais.
    raw : (HW,)
        Energia bruta (antes do reshape e suavizamento) por localização.
    grid : (H, W)
        Dimensões espaciais da grade de ativação.
    coords_2d : (HW, 2)
        Coordenadas espectrais 2D (√λ v) no espaço de ativações.
    coords_3d : (HW, 3)
        Coordenadas espectrais 3D (√λ v) no espaço de ativações.
    eigenvalues : (C,)
        Autovalores da Gram em ordem decrescente.
    """

    field: torch.Tensor
    raw: torch.Tensor
    grid: tuple[int, int]
    coords_2d: torch.Tensor
    coords_3d: torch.Tensor
    eigenvalues: torch.Tensor


@dataclass
class GramMetricUsage:
    """Uso efetivo da métrica Gram no campo de ativações.

    Campos
    ------
    surface : GramEnergySurface
        Superfície de energia induzida pela Gram.
    used_similarity : (C, C)
        Produto elemento a elemento entre a Gram e a covariância empírica do
        campo centrado — mede quais pares de canais a Gram efetivamente pondera.
    channel_usage : (C,)
        Soma por linha de ``used_similarity``; indica o quanto cada canal
        contribui para a energia induzida.
    """

    surface: GramEnergySurface
    used_similarity: torch.Tensor
    channel_usage: torch.Tensor


def activation_to_spatial_field(activation: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    if activation.ndim == 4:
        activation = activation[0]
    if activation.ndim != 3:
        raise ValueError("Ativação espacial deve ter shape (C,H,W) ou (1,C,H,W).")
    grid = (int(activation.shape[1]), int(activation.shape[2]))
    return activation.detach().float().flatten(1).T, grid


def _gaussian_blur(field: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return field
    radius = max(1, int(4 * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, dtype=field.dtype, device=field.device)
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    pad_h = min(radius, max(int(field.shape[-2]) - 1, 0))
    pad_w = min(radius, max(int(field.shape[-1]) - 1, 0))
    img = field.unsqueeze(0).unsqueeze(0)
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
    kh = kernel.view(1, 1, -1, 1)
    kw = kernel.view(1, 1, 1, -1)
    if pad_h != radius:
        kh = kh[:, :, radius - pad_h : radius + pad_h + 1, :]
        kh = kh / kh.sum()
    if pad_w != radius:
        kw = kw[:, :, :, radius - pad_w : radius + pad_w + 1]
        kw = kw / kw.sum()
    return F.conv2d(F.conv2d(img, kh), kw)[0, 0]


def gram_induced_energy_surface(
    gram: torch.Tensor,
    field: torch.Tensor,
    grid: tuple[int, int],
    center: bool = True,
    power: float = 1.0,
    smooth_sigma: float | None = 1.0,
) -> GramEnergySurface:
    field = field.detach().float()
    gram = gram.detach().to(device=field.device, dtype=field.dtype)
    gram = 0.5 * (gram + gram.T)
    if center:
        field = field - field.mean(dim=0, keepdim=True)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    eigenvectors = eigenvectors[:, order]
    coeffs = field @ eigenvectors
    raw = (coeffs.square() * eigenvalues.pow(power).unsqueeze(0)).sum(dim=1)
    mapped = raw.reshape(grid)
    if smooth_sigma is not None:
        mapped = _gaussian_blur(mapped, smooth_sigma)
    coords = coeffs * eigenvalues.pow(power / 2).unsqueeze(0)
    return GramEnergySurface(
        field=mapped,
        raw=raw,
        grid=grid,
        coords_2d=F.pad(coords[:, :2], (0, max(0, 2 - coords[:, :2].shape[1]))),
        coords_3d=F.pad(coords[:, :3], (0, max(0, 3 - coords[:, :3].shape[1]))),
        eigenvalues=eigenvalues,
    )


def gram_metric_usage(gram: torch.Tensor, field: torch.Tensor, grid: tuple[int, int]) -> GramMetricUsage:
    channels = min(int(gram.shape[0]), int(field.shape[1]))
    field = field.detach().float()[:, :channels]
    gram = gram.detach().to(device=field.device, dtype=field.dtype)[:channels, :channels]
    surface = gram_induced_energy_surface(gram, field, grid)
    centered = field - field.mean(dim=0, keepdim=True)
    used_similarity = (centered.T @ centered) * gram
    used_similarity = 0.5 * (used_similarity + used_similarity.T)
    return GramMetricUsage(surface, used_similarity, used_similarity.abs().sum(dim=1))


def psd_positive_part(matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.detach().float()
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    return (eigenvectors * eigenvalues.clamp_min(0).unsqueeze(0)) @ eigenvectors.T


def normalized_delta(
    concept_usage: torch.Tensor, 
    contrast_usage: torch.Tensor | None, 
    eps: float = 1e-8
) -> torch.Tensor:
    if contrast_usage is None:
        return concept_usage
    contrast_usage = contrast_usage.to(device=concept_usage.device, dtype=concept_usage.dtype)
    return (concept_usage - contrast_usage) / (concept_usage.abs() + contrast_usage.abs() + eps)


def suppress_borders(score_map: torch.Tensor, fraction: float) -> torch.Tensor:
    if fraction <= 0:
        return score_map
    h, w = int(score_map.shape[0]), int(score_map.shape[1])
    bh = min(int(round(h * fraction)), max(h // 2 - 1, 0))
    bw = min(int(round(w * fraction)), max(w // 2 - 1, 0))
    out = score_map.clone()
    if bh > 0:
        out[:bh, :] = 0
        out[-bh:, :] = 0
    if bw > 0:
        out[:, :bw] = 0
        out[:, -bw:] = 0
    return out


def orthogonal_energy_surface(
    gram: torch.Tensor,
    field: torch.Tensor,
    grid: tuple[int, int],
    tau: float = 0.95,
    energy_kind: str = "operator",
    center: bool = True,
    smooth_sigma: float | None = 1.0,
) -> OrthogonalEnergySurface:
    """Mapa de energia residual no complemento ortogonal V_r⊥ do núcleo.

    Para cada localização espacial s com vetor de ativação x_s ∈ ℝ^C, computa:

        E_perp(s) = ‖P_⊥ x_s‖²  =  ‖x_s‖² − ‖V_r^T x_s‖²
        R(s)      = E_perp(s) / ‖x_s‖²  ∈ [0, 1]

    onde V_r são os r autovetores principais do gram K que capturam a fração
    tau de energia (critério "operator" = Σλ², ou "trace" = Σλ).

    Regiões com R(s) alto são 'pontos cegos' do núcleo: as ativações nessas
    localizações projetam-se majoritariamente em direções que o subespaço
    principal não cobre até o limiar tau. O núcleo não as distingue.

    Parâmetros
    ----------
    gram : (C, C)
        Matriz de Gram do espaço de tipos (tipicamente K_total).
    field : (HW, C)
        Campo de ativações espaciais (saída de activation_to_spatial_field).
    grid : (H, W)
        Dimensões espaciais correspondentes ao field.
    tau : float
        Limiar de energia para selecionar V_r (padrão 0.95).
    energy_kind : "operator" | "trace"
        Critério de energia para selecionar rank r (padrão "operator" = Σλ²).
    center : bool
        Se True, subtrai a média espacial antes de projetar (padrão True).
    smooth_sigma : float | None
        Desvio padrão do suavizamento gaussiano aplicado aos mapas resultantes.
        None = sem suavizamento.
    """
    from ..gram.eigenspace import principal_subspace

    # field determina o device alvo; gram é movido para ele (segue o padrão de
    # gram_metric_usage e gram_induced_energy_surface: gram → field.device).
    field = field.detach().float()
    gram = gram.detach().to(device=field.device, dtype=field.dtype)
    gram = 0.5 * (gram + gram.T)

    if center:
        field = field - field.mean(dim=0, keepdim=True)

    # Subespaço principal V_r: colunas são os r autovetores principais (C, r)
    sub = principal_subspace(gram, tau=tau, energy=energy_kind)
    V_r = sub.basis  # (C, r) — mesmo device e dtype de gram/field

    # Projeção de cada localização s em V_r: (HW, r)
    proj = field @ V_r          # (HW, r)

    # Energia em V_r e em V_r⊥ por localização
    energy_in_Vr = (proj * proj).sum(dim=1)                      # (HW,)
    total_energy = (field * field).sum(dim=1)                    # (HW,)
    perp_energy = (total_energy - energy_in_Vr).clamp_min(0.0)  # (HW,)
    perp_ratio = perp_energy / total_energy.clamp_min(1e-12)     # (HW,)

    # Reshape para grade espacial
    perp_energy_map = perp_energy.reshape(grid)
    perp_ratio_map = perp_ratio.reshape(grid)
    total_energy_map = total_energy.reshape(grid)

    if smooth_sigma is not None and smooth_sigma > 0:
        perp_energy_map = _gaussian_blur(perp_energy_map, smooth_sigma)
        perp_ratio_map = _gaussian_blur(perp_ratio_map, smooth_sigma)
        total_energy_map = _gaussian_blur(total_energy_map, smooth_sigma)

    return OrthogonalEnergySurface(
        perp_energy=perp_energy_map,
        perp_ratio=perp_ratio_map,
        total_energy=total_energy_map,
        grid=grid,
        r=sub.r,
        tau=tau,
        coverage=float(sub.captured_fraction()),
    )


def metric_score_and_coords(
    metric: torch.Tensor, 
    field: torch.Tensor, 
    grid: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    field = field.detach().float()
    metric = metric.detach().to(device=field.device, dtype=field.dtype)
    field = field - field.mean(dim=0, keepdim=True)
    metric = 0.5 * (metric + metric.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(metric)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    eigenvectors = eigenvectors[:, order]
    score = (field @ metric * field).sum(dim=1).clamp_min(0).reshape(grid)
    coords = field @ eigenvectors[:, :3]
    coords = coords * eigenvalues[:3].sqrt().unsqueeze(0)
    return score, coords

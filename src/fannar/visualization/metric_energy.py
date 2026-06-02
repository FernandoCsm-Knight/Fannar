from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from fannar.concepts.metric_energy import GramEnergySurface, OrthogonalEnergySurface, suppress_borders


def image_to_hwc(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    return image.detach().cpu().clamp(0, 1)


def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_image_energy_panel(
    image: torch.Tensor,
    surface: GramEnergySurface | torch.Tensor,
    title: str,
    path: Path,
    contours: bool = True,
) -> None:
    field = surface.field if isinstance(surface, GramEnergySurface) else surface
    img = image_to_hwc(image)
    h, w = int(img.shape[0]), int(img.shape[1])
    heat = F.interpolate(field.reshape(1, 1, *field.shape).float(), size=(h, w), mode="bilinear",
                         align_corners=False)[0, 0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), gridspec_kw={"width_ratios": [1, 1.08]})
    axes[0].imshow(img.numpy())
    axes[0].set_title("Imagem original")
    axes[0].axis("off")
    hm = axes[1].imshow(heat.detach().cpu().numpy(), cmap="magma")
    if contours:
        axes[1].contour(heat.detach().cpu().numpy(), levels=12, colors="white", linewidths=0.45, alpha=0.6)
    axes[1].set_title(title)
    axes[1].axis("off")
    fig.colorbar(hm, ax=axes[1], fraction=0.046, pad=0.04)
    save_current_figure(path)


def plot_orthogonal_energy_panel(
    image: torch.Tensor,
    surface: OrthogonalEnergySurface,
    title: str,
    path: Path,
    use_ratio: bool = True,
    contours: bool = True,
    autoscale: bool = True,
) -> None:
    """Painel de energia no complemento ortogonal V_r⊥.

    Visualiza onde no espaço espacial as ativações NÃO são capturadas pelo
    subespaço principal do núcleo. Segue o mesmo estilo de plot_image_energy_panel.

    Parâmetros
    ----------
    use_ratio : bool
        True (padrão): plota a razão R(s) = E_perp(s)/E_total(s).
        False: plota a energia absoluta E_perp(s).
    autoscale : bool
        True (padrão): escala a cor ao intervalo real dos dados [min, max],
        maximizando o contraste. False: fixa vmin=0, vmax=1 (escala absoluta,
        útil para comparar camadas na mesma escala).
    """
    field = surface.perp_ratio if use_ratio else surface.perp_energy
    img = image_to_hwc(image)
    h, w = int(img.shape[0]), int(img.shape[1])
    heat = F.interpolate(
        field.reshape(1, 1, *field.shape).float(),
        size=(h, w), mode="bilinear", align_corners=False,
    )[0, 0]
    heat_np = heat.detach().cpu().numpy()

    cmap = "viridis"
    ylabel = "Razão ortogonal R(s)" if use_ratio else "Energia ortogonal E_⊥(s)"
    vmin = None if autoscale else 0.0
    vmax = None if autoscale else (1.0 if use_ratio else None)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), gridspec_kw={"width_ratios": [1, 1.08]})
    axes[0].imshow(img.numpy())
    axes[0].set_title("Imagem original")
    axes[0].axis("off")

    hm = axes[1].imshow(heat_np, cmap=cmap, vmin=vmin, vmax=vmax)
    if contours:
        axes[1].contour(heat_np, levels=10, colors="white", linewidths=0.45, alpha=0.5)
    tag = f"r={surface.r}, cov={surface.coverage:.2f}, tau={surface.tau}"
    axes[1].set_title(f"{title}\n{tag}", fontsize=9)
    axes[1].axis("off")
    cb = fig.colorbar(hm, ax=axes[1], fraction=0.046, pad=0.04)
    cb.set_label(ylabel, fontsize=8)
    save_current_figure(path)


def crop_patch(
    image: torch.Tensor, 
    score_map: torch.Tensor, 
    crop_size: int, 
    border_fraction: float
) -> torch.Tensor:
    image = image.detach().cpu()
    h, w = int(image.shape[1]), int(image.shape[2])
    score_map = suppress_borders(score_map, border_fraction)
    up = F.interpolate(score_map.reshape(1, 1, *score_map.shape).float(), size=(h, w), mode="bilinear",
                       align_corners=False)[0, 0]
    cy, cx = divmod(int(up.flatten().argmax()), w)
    half = crop_size // 2
    padded = F.pad(image.unsqueeze(0), (half, half, half, half), mode="reflect")[0]
    return padded[:, cy : cy + crop_size, cx : cx + crop_size]


def plot_patch_montage(patches: list[torch.Tensor], title: str, path: Path) -> None:
    if not patches:
        return
    ncols = min(6, len(patches))
    nrows = (len(patches) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows))
    if nrows == 1 and ncols == 1:
        axes_iter = [axes]
    elif nrows == 1 or ncols == 1:
        axes_iter = list(axes)
    else:
        axes_iter = [ax for row in axes for ax in row]
    for ax, patch in zip(axes_iter, patches, strict=False):
        ax.imshow(image_to_hwc(patch).numpy())
        ax.axis("off")
    for ax in axes_iter[len(patches):]:
        ax.axis("off")
    fig.suptitle(title)
    save_current_figure(path)

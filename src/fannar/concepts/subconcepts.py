from __future__ import annotations

import torch
import torch.nn.functional as F

from .metric_energy import activation_to_spatial_field, metric_score_and_coords, suppress_borders


def crop_spatial_patch(
    image: torch.Tensor,
    row: int,
    col: int,
    grid: tuple[int, int],
    crop_size: int,
) -> torch.Tensor:
    image = image.detach().cpu()
    h, w = int(image.shape[1]), int(image.shape[2])
    cy = int(round((row + 0.5) * h / max(grid[0], 1)))
    cx = int(round((col + 0.5) * w / max(grid[1], 1)))
    half = crop_size // 2
    padded = F.pad(image.unsqueeze(0), (half, half, half, half), mode="reflect")[0]
    return padded[:, cy : cy + crop_size, cx : cx + crop_size]


def collect_subconcept_records(
    activations: torch.Tensor,
    metric: torch.Tensor,
    border_fraction: float = 0.0,
) -> list[tuple[int, int, int, tuple[int, int], float]]:
    records: list[tuple[int, int, int, tuple[int, int], float]] = []
    for sample_idx in range(int(activations.shape[0])):
        field, grid = activation_to_spatial_field(activations[sample_idx])
        channels = min(int(metric.shape[0]), int(field.shape[1]))
        score, _coords = metric_score_and_coords(metric[:channels, :channels], field[:, :channels], grid)
        score = suppress_borders(score, border_fraction)
        flat = score.flatten()
        for flat_idx in torch.nonzero(flat > 0).flatten().tolist():
            row, col = divmod(int(flat_idx), grid[1])
            records.append((sample_idx, row, col, grid, float(flat[flat_idx])))
    return records


def top_metric_patches(
    images: torch.Tensor,
    activations: torch.Tensor,
    metric: torch.Tensor,
    max_patches: int = 18,
    crop_size: int = 48,
    border_fraction: float = 0.0,
    min_quantile: float = 0.90,
    max_per_image: int = 3,
) -> list[torch.Tensor]:
    records = collect_subconcept_records(activations, metric, border_fraction=border_fraction)
    if not records:
        return []

    scores = torch.tensor([record[4] for record in records], dtype=torch.float32)
    threshold = float(torch.quantile(scores, min_quantile))
    records = [record for record in records if record[4] >= threshold]
    records = sorted(records, key=lambda record: record[4], reverse=True)

    patches: list[torch.Tensor] = []
    per_image: dict[int, int] = {}
    for record in records:
        sample_index, row, col, grid, _score = record
        used = per_image.get(sample_index, 0)
        if used >= max_per_image:
            continue
        patches.append(
            crop_spatial_patch(
                images[sample_index],
                row,
                col,
                grid,
                crop_size,
            )
        )
        per_image[sample_index] = used + 1
        if len(patches) >= max_patches:
            break
    return patches

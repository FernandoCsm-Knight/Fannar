"""Artefatos: empacota geometrias de camada/modelo em um diretório."""

from __future__ import annotations

from pathlib import Path

import torch

from ..types import LayerGeometry


def save_layer_geometry(geo: LayerGeometry, path: str | Path) -> None:
    """Salva uma geometria de camada (tensores em CPU)."""
    payload = {
        "K_total": geo.K_total.cpu(),
        "D": geo.D.cpu(),
        "eigenvalues": geo.spectrum.eigenvalues.cpu(),
        "eigenvectors": geo.spectrum.eigenvectors.cpu(),
        "coords_2d": geo.coords_2d.cpu(),
        "coords_3d": geo.coords_3d.cpu(),
        "coverage": geo.coverage.detail,
        "profile": geo.profile.as_dict() if geo.profile else None,
        "labels": geo.labels,
        "metadata": geo.metadata,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_layer_payload(path: str | Path, map_location: str | None = None) -> dict:
    return torch.load(str(path), map_location=map_location, weights_only=False)

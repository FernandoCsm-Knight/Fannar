"""Serialização de matrizes de Gram e geometrias via torch.save/load."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..gram.matrix import GramMatrix


def save_gram(gram: GramMatrix, path: str | Path) -> None:
    payload: dict[str, Any] = {
        "values": gram.values.cpu(),
        "labels": gram.labels,
        "axis_type": gram.axis_type,
        "metadata": gram.metadata,
    }
    torch.save(payload, str(path))


def load_gram(path: str | Path, map_location: str | None = None) -> GramMatrix:
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    return GramMatrix(
        values=payload["values"],
        labels=payload.get("labels"),
        axis_type=payload.get("axis_type", "features"),
        metadata=payload.get("metadata", {}),
    )

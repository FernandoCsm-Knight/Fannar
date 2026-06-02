"""Utilitários de device e dtype."""

from __future__ import annotations

import torch

from ..config import get_config


def resolve_device(
    *tensors: torch.Tensor, device: torch.device | str | None = None
) -> torch.device:
    """Resolve o device alvo.

    Prioridade: argumento ``device`` > config global > device do primeiro
    tensor > CPU.
    """
    if device is not None:
        return torch.device(device)
    cfg = get_config()
    if cfg.device is not None:
        return cfg.device
    for t in tensors:
        if isinstance(t, torch.Tensor):
            return t.device
    return torch.device("cpu")


def resolve_dtype(
    *tensors: torch.Tensor, dtype: torch.dtype | None = None
) -> torch.dtype:
    """Resolve o dtype alvo (argumento > primeiro tensor float > config)."""
    if dtype is not None:
        return dtype
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            return t.dtype
    return get_config().dtype


def to_target(
    t: torch.Tensor,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Move ``t`` para device/dtype resolvidos."""
    dev = resolve_device(t, device=device)
    dt = resolve_dtype(t, dtype=dtype)
    return t.to(device=dev, dtype=dt)


def cuda_available() -> bool:
    return torch.cuda.is_available()

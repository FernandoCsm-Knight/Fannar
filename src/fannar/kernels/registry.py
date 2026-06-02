"""Registro de kernels por nome, para construção via configuração/serialização."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..errors import ConfigurationError
from ..types import Kernel
from .cosine import CosineKernel
from .linear import LinearKernel
from .polynomial import PolynomialKernel
from .rbf import RBFKernel

_REGISTRY: dict[str, Callable[..., Kernel]] = {
    "linear": LinearKernel,
    "cosine": CosineKernel,
    "rbf": RBFKernel,
    "polynomial": PolynomialKernel,
    "poly": PolynomialKernel,
}


def register_kernel(name: str, factory: Callable[..., Kernel]) -> None:
    """Registra uma fábrica de kernel sob ``name``."""
    _REGISTRY[name.lower()] = factory


def get_kernel(name: str, **kwargs: Any) -> Kernel:
    """Instancia um kernel registrado pelo nome."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ConfigurationError(
            f"Kernel '{name}' desconhecido. Disponíveis: {sorted(_REGISTRY)}."
        )
    return _REGISTRY[key](**kwargs)


def available_kernels() -> list[str]:
    return sorted(_REGISTRY)

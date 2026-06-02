"""Configuração global da fannar.

Centraliza tolerâncias numéricas e padrões de dtype/device. Todas as funções
numéricas aceitam ``eps`` explícito; este módulo apenas fornece os defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch


@dataclass(frozen=True)
class FannarConfig:
    """Parâmetros padrão usados quando não especificados explicitamente.

    Attributes
    ----------
    eps:
        Tolerância numérica geral (divisões, comparações de simetria).
    eps_psd:
        Limiar abaixo do qual autovalores negativos são considerados ruído
        e clampados a zero.
    dtype:
        dtype padrão para cálculos. ``float64`` é recomendado para estabilidade
        de decomposições espectrais; ``float32`` para grandes volumes.
    device:
        device padrão. ``None`` significa "respeitar o device do tensor de
        entrada".
    strict:
        Quando ``True``, validações lançam exceção em vez de corrigir
        silenciosamente.
    max_gram_n:
        Limite de ``n`` acima do qual operações densas O(n^2) ou O(n^3)
        emitem aviso (não bloqueiam).
    """

    eps: float = 1e-12
    eps_psd: float = 1e-8
    dtype: torch.dtype = torch.float64
    device: torch.device | None = None
    strict: bool = False
    max_gram_n: int = 8192
    extra: dict[str, Any] = field(default_factory=dict)

    def update(self, **kwargs: Any) -> FannarConfig:
        """Retorna uma cópia com campos sobrescritos."""
        return replace(self, **kwargs)


# Config global mutável por referência (substituível por set_config).
_GLOBAL_CONFIG = FannarConfig()


def get_config() -> FannarConfig:
    """Retorna a configuração global atual."""
    return _GLOBAL_CONFIG


def set_config(config: FannarConfig) -> None:
    """Substitui a configuração global."""
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = config

"""Base de kernels.

Convenção de shapes: ``x`` tem shape ``(n, d)``, ``y`` (opcional) ``(m, d)``.
O retorno é a matriz de Gram ``(n, m)`` (ou ``(n, n)`` quando ``y is None``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from ..errors import ShapeError


class BaseKernel(ABC):
    """Classe base para kernels com validação de shape e simetrização opcional."""

    name: str = "base"

    @abstractmethod
    def _compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Cálculo bruto da Gram entre ``x (n,d)`` e ``y (m,d)``."""

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self._as_2d(x)
        symmetric = y is None
        y2 = x if y is None else self._as_2d(y)
        if x.shape[1] != y2.shape[1]:
            raise ShapeError(
                f"Dimensão de feature incompatível: x tem d={x.shape[1]}, "
                f"y tem d={y2.shape[1]}."
            )
        K = self._compute(x, y2)
        if symmetric:
            K = 0.5 * (K + K.T)
        return K

    @staticmethod
    def _as_2d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return x.unsqueeze(0)
        if x.ndim == 2:
            return x
        # achata dimensões de feature: (n, ...) -> (n, prod(...))
        return x.reshape(x.shape[0], -1)

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return f"{self.__class__.__name__}()"

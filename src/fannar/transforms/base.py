"""Base de transformações pós-kernel ``T: K -> K``.

Toda transformação admissível (Documento da teoria, Def. de pipeline
núcleo-transformação) é equivariante por permutação dos objetos e preserva,
sempre que possível, simetria e PSD.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from ..utils.linalg import symmetrize


class BaseTransform(ABC):
    """Classe base. Simetriza a entrada por padrão e delega o cálculo."""

    name: str = "base"
    symmetrize_input: bool = True

    @abstractmethod
    def _apply(self, K: torch.Tensor) -> torch.Tensor:
        ...

    def __call__(self, K: torch.Tensor) -> torch.Tensor:
        if self.symmetrize_input:
            K = symmetrize(K)
        return self._apply(K)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}()"

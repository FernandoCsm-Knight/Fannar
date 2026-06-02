"""Pipeline núcleo-transformação ``(κ, T)``.

Produz a Gram bruta via kernel e aplica a transformação pós-kernel admissível.
"""

from __future__ import annotations

import torch

from ..gram.matrix import GramMatrix
from ..transforms.centering import IdentityTransform
from ..types import Kernel, KernelTransform


class KernelPipeline:
    """Encapsula ``K = T(κ(X))``."""

    def __init__(self, kernel: Kernel, transform: KernelTransform | None = None) -> None:
        self.kernel = kernel
        self.transform = transform or IdentityTransform()

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        labels: list[str] | None = None,
        axis_type: str = "features",
    ) -> GramMatrix:
        K_raw = self.kernel(x, y)
        K = self.transform(K_raw)
        return GramMatrix(values=K, labels=labels, axis_type=axis_type,
                          metadata={"kernel": getattr(self.kernel, "name", repr(self.kernel))})

    def gram(self, x: torch.Tensor, **kwargs) -> GramMatrix:
        return self(x, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover
        return f"KernelPipeline({self.kernel!r}, {self.transform!r})"

"""A classe central :class:`GramMatrix`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ..types import AxisType, EnergyKind, PrincipalSubspace, SpectralDecomposition
from ..utils.linalg import symmetrize
from ..utils.validation import check_psd, check_symmetric, warn_large
from . import eigenspace as _eig
from .distance import distance_matrix
from .psd import project_psd


@dataclass
class GramMatrix:
    """Matriz de Gram com metadados e operações geométricas.

    Attributes
    ----------
    values:
        Tensor ``(n, n)``.
    labels:
        Rótulos opcionais dos ``n`` objetos.
    axis_type:
        Tipo dos objetos ("channels", "tokens", "patches", "features", ...).
    metadata:
        Dicionário livre.
    """

    values: torch.Tensor
    labels: list[str] | None = None
    axis_type: AxisType = "features"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.shape[0] != self.values.shape[1]:
            raise ValueError(f"values deve ser (n,n), recebido {tuple(self.values.shape)}.")
        if self.labels is not None and len(self.labels) != self.n:
            raise ValueError(
                f"labels tem {len(self.labels)} entradas, esperado {self.n}."
            )

    # --- propriedades básicas ---
    @property
    def n(self) -> int:
        return int(self.values.shape[0])

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def dtype(self) -> torch.dtype:
        return self.values.dtype

    # --- validação e correção ---
    def validate(self, check_psd_flag: bool = True, strict: bool | None = None) -> GramMatrix:
        """Valida simetria e (opcionalmente) PSD. Retorna ``self`` para encadear."""
        warn_large(self.n, "GramMatrix.validate")
        check_symmetric(self.values, name="GramMatrix", strict=strict)
        if check_psd_flag:
            check_psd(self.values, name="GramMatrix", strict=strict)
        return self

    def symmetrize(self) -> GramMatrix:
        return self._with(symmetrize(self.values))

    def nearest_psd(self, eps: float = 1e-8) -> GramMatrix:
        return self._with(project_psd(self.values, eps=eps))

    # --- operações geométricas ---
    def distance_matrix(self, clamp: bool = True, squared: bool = False) -> torch.Tensor:
        return distance_matrix(self.values, clamp=clamp, squared=squared)

    def eigendecompose(self, descending: bool = True) -> SpectralDecomposition:
        return _eig.eigendecompose(self.values, descending=descending)

    def spectral_coordinates(self, dim: int = 2) -> torch.Tensor:
        return _eig.spectral_coordinates(self.values, dim=dim)

    def embedding_fidelity(self, dim: int = 2) -> float:
        return _eig.embedding_fidelity(self.values, dim=dim)

    def principal_subspace(
        self, tau: float = 0.95, energy: EnergyKind = "operator"
    ) -> PrincipalSubspace:
        return _eig.principal_subspace(self.values, tau=tau, energy=energy)

    # --- combinação ---
    def hadamard(self, other: GramMatrix | torch.Tensor) -> GramMatrix:
        ov = other.values if isinstance(other, GramMatrix) else other
        return self._with(symmetrize(self.values * ov))

    # --- helpers ---
    def to(self, device=None, dtype=None) -> GramMatrix:
        return self._with(self.values.to(device=device, dtype=dtype))

    def _with(self, values: torch.Tensor) -> GramMatrix:
        return GramMatrix(
            values=values,
            labels=self.labels,
            axis_type=self.axis_type,
            metadata=dict(self.metadata),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GramMatrix(n={self.n}, axis_type='{self.axis_type}', "
            f"dtype={self.dtype}, device={self.device})"
        )

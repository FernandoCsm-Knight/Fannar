"""Tipos e dataclasses de resultado da fannar.

Convenção central de shapes
---------------------------
- Uma matriz de Gram ``K`` tem shape ``(n, n)`` onde ``n`` é o número de
  objetos analisados (canais, tokens, patches, conceitos...).
- Um conjunto de objetos a comparar é representado por ``X`` de shape
  ``(n, d)`` onde ``d`` é a dimensão de feature de cada objeto.
- Autovetores em ``SpectralDecomposition.eigenvectors`` têm shape ``(n, n)``,
  com as COLUNAS sendo os autovetores (coluna ``k`` = ``v_k``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import torch

Metric = Literal["euclidean", "rkhs"]
EnergyKind = Literal["operator", "trace"]
AxisType = str  # "channels" | "tokens" | "patches" | "features" | "concepts" | ...


@runtime_checkable
class Kernel(Protocol):
    """Interface comum de kernels. Produz uma matriz de Gram bruta."""

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        ...


@runtime_checkable
class KernelTransform(Protocol):
    """Interface comum de transformações pós-kernel ``K -> K``."""

    def __call__(self, K: torch.Tensor) -> torch.Tensor:
        ...


@dataclass
class SpectralDecomposition:
    """Resultado de uma decomposição espectral simétrica ``K = V Λ Vᵀ``.

    eigenvalues:
        Autovalores ``(n,)``, ordenados de forma decrescente por padrão.
    eigenvectors:
        Autovetores como COLUNAS ``(n, n)``. ``eigenvectors[:, k]`` é ``v_k``.
    """

    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    descending: bool = True

    @property
    def n(self) -> int:
        return int(self.eigenvalues.shape[0])

    def reconstruct(self) -> torch.Tensor:
        """Reconstrói ``K = V Λ Vᵀ``."""
        return (self.eigenvectors * self.eigenvalues.unsqueeze(0)) @ self.eigenvectors.T

    def trace_energy(self) -> torch.Tensor:
        return self.eigenvalues.clamp_min(0).sum()

    def operator_energy(self) -> torch.Tensor:
        return (self.eigenvalues.clamp_min(0) ** 2).sum()


@dataclass
class PrincipalSubspace:
    """Subespaço principal ``V_r`` retido a partir de um limiar de energia.

    eigenvalues / eigenvectors:
        Espectro completo (referência), ordenado decrescente.
    r:
        Número de modos retidos.
    tau:
        Limiar de energia usado.
    energy:
        "operator" (Σλ²) ou "trace" (Σλ).
    """

    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    r: int
    tau: float
    energy: EnergyKind = "operator"

    @property
    def n(self) -> int:
        return int(self.eigenvalues.shape[0])

    @property
    def basis(self) -> torch.Tensor:
        """Autovetores que geram ``V_r`` como colunas ``(n, r)``."""
        return self.eigenvectors[:, : self.r]

    @property
    def basis_perp(self) -> torch.Tensor:
        """Autovetores que geram ``V_r^perp`` como colunas ``(n, n-r)``."""
        return self.eigenvectors[:, self.r :]

    def captured_fraction(self) -> torch.Tensor:
        """Fração de energia retida por ``V_r`` segundo o critério ``energy``."""
        lam = self.eigenvalues.clamp_min(0)
        w = lam ** 2 if self.energy == "operator" else lam
        total = w.sum()
        return w[: self.r].sum() / total.clamp_min(torch.finfo(w.dtype).tiny)


@dataclass
class EnergyDecomposition:
    """Decomposição de energia de uma função/vetor ``u`` em ``V_r`` ⊕ ``V_r^perp``."""

    total: torch.Tensor
    captured: torch.Tensor
    residual: torch.Tensor
    residual_ratio: torch.Tensor
    parallel: torch.Tensor  # componente em V_r (mesmo espaço de u)
    orthogonal: torch.Tensor  # componente em V_r^perp


@dataclass
class DiagnosticResult:
    """Resultado genérico de uma métrica diagnóstica.

    Usa ``available`` para sinalizar métricas indefinidas (ex.: sem rótulos),
    evitando propagar ``NaN``.
    """

    name: str
    value: torch.Tensor | float | None
    available: bool = True
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplanatoryProfile:
    """Perfil explicativo de oito eixos (modelos de explicação científica)."""

    DN: DiagnosticResult
    IS: DiagnosticResult
    RE: DiagnosticResult
    Pr: DiagnosticResult
    Un: DiagnosticResult
    CM: DiagnosticResult
    NM: DiagnosticResult
    In: DiagnosticResult

    AXES = ("DN", "IS", "RE", "Pr", "Un", "CM", "NM", "In")

    def as_vector(self, fill: float = 0.0) -> torch.Tensor:
        """Vetor ``(8,)`` na ordem canônica; eixos indisponíveis recebem ``fill``."""
        vals = []
        for ax in self.AXES:
            res: DiagnosticResult = getattr(self, ax)
            if res.available and res.value is not None:
                vals.append(float(res.value))
            else:
                vals.append(fill)
        return torch.tensor(vals)

    def as_dict(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for ax in self.AXES:
            res: DiagnosticResult = getattr(self, ax)
            out[ax] = float(res.value) if (res.available and res.value is not None) else None
        return out


@dataclass
class LayerGeometry:
    """Geometria completa de uma camada (saída de LayerPipeline.analyze).

    Os quatro indicadores intra-camada da teoria (main.tex, subseção 4.1) são:
    - ``dispersion``: Disp_ℓ — dispersão estrutural (distância média entre canais).
    - ``stability``:  Stab_ℓ — estabilidade local (distância k-NN por canal).
    - ``coverage``:   Cov_ℓ  — cobertura espectral (fração de modos retidos).
    - ``residual``:   Res_ℓ  — energia residual de uma direção de interesse.
    """

    K_W: torch.Tensor | None
    K_A: torch.Tensor | None
    K_G: torch.Tensor | None
    K_total: torch.Tensor
    D: torch.Tensor
    spectrum: SpectralDecomposition
    coords_2d: torch.Tensor
    coords_3d: torch.Tensor
    coverage: DiagnosticResult
    dispersion: DiagnosticResult | None = None
    stability: DiagnosticResult | None = None
    residual: DiagnosticResult | None = None
    profile: ExplanatoryProfile | None = None
    labels: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

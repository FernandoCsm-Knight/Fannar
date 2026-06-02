"""Pipeline de camada: combina K_W, K_A, K_G e produz a geometria completa."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..diagnostics.coverage import coverage
from ..diagnostics.dispersion import dispersion as compute_dispersion
from ..diagnostics.explanatory_profile import explanatory_profile
from ..diagnostics.residual import residual_energy
from ..diagnostics.stability import stability as compute_stability
from ..gram.distance import distance_matrix
from ..gram.eigenspace import eigendecompose, spectral_coordinates
from ..gram.tensor import TensorGram
from ..transforms.normalization import TraceNormalizeTransform
from ..types import EnergyKind, LayerGeometry
from .kernel_pipeline import KernelPipeline

_trace_norm = TraceNormalizeTransform()


@dataclass
class LayerRepresentations:
    """Representações brutas de uma camada (entrada do LayerPipeline).

    Cada campo é um tensor ``(C, d)`` (C objetos = canais; d = dimensão da
    feature daquela fonte) ou ``None`` se a fonte não for usada.
    """

    weights: torch.Tensor | None = None
    activations: torch.Tensor | None = None
    gradients: torch.Tensor | None = None
    labels: list[str] | None = None


class LayerPipeline:
    """Combina até três pipelines de fonte (pesos, ativações, gradientes)."""

    def __init__(
        self,
        weights: KernelPipeline | None = None,
        activations: KernelPipeline | None = None,
        gradients: KernelPipeline | None = None,
    ) -> None:
        self.weights = weights
        self.activations = activations
        self.gradients = gradients

    def analyze(
        self,
        reps: LayerRepresentations,
        tau: float = 0.95,
        energy: EnergyKind = "operator",
        gradient_direction: torch.Tensor | None = None,
        other_layer_grams: Sequence[torch.Tensor] | None = None,
        other_layer_distances: Sequence[torch.Tensor] | None = None,
        eta_H: float = 1.0,
        eta_W: float = 1.0,
        k_stab: int = 5,
        compute_profile: bool = True,
        normalize_total: bool = True,
    ) -> LayerGeometry:
        """Analisa uma camada e retorna sua geometria completa.

        Parâmetros
        ----------
        other_layer_distances:
            Matrizes de distâncias de outras camadas. Se fornecido, α_Un usa a
            fórmula exata da teoria (entropia espectral do laplaciano + Wasserstein).
        eta_H, eta_W:
            Escalas de normalização para Sim(ℓ, ℓ'). Calibrar como mediana das
            discrepâncias inter-camada sobre o conjunto de referência.
        k_stab:
            Número de vizinhos para o indicador de estabilidade local.
        normalize_total:
            Se ``True`` (padrão), aplica TraceNorm em ``K_total`` após o produto
            de Hadamard. Necessário porque o Hadamard de d matrizes com trace=1
            e valores O(1/C) produz K_total com trace O((1/C)^{d-1}), tornando
            as distâncias resultantes quasi-nulas e a afinidade gaussiana uniforme
            (o que colapsa CM e NM para zero por aritmética, não por geometria).
            A normalização final preserva a estrutura multiplicativa (zeros
            permanecem zeros) e restaura a escala para análise geométrica.
        """
        grams = []
        K_W = K_A = K_G = None
        if self.weights is not None and reps.weights is not None:
            K_W = self.weights(reps.weights, axis_type="channels").values
            grams.append(K_W)
        if self.activations is not None and reps.activations is not None:
            K_A = self.activations(reps.activations, axis_type="channels").values
            grams.append(K_A)
        if self.gradients is not None and reps.gradients is not None:
            K_G = self.gradients(reps.gradients, axis_type="channels").values
            grams.append(K_G)
        if not grams:
            raise ValueError("Nenhuma fonte fornecida para combinar.")

        K_total = TensorGram.combine(grams).values
        if normalize_total:
            K_total = _trace_norm(K_total)
        D = distance_matrix(K_total)
        spectrum = eigendecompose(K_total, descending=True)
        coords_2d = spectral_coordinates(K_total, dim=2)
        coords_3d = spectral_coordinates(K_total, dim=3)
        cov = coverage(K_total, tau=tau, energy=energy)

        # Quatro indicadores intra-camada (main.tex, subseção 4.1)
        disp = compute_dispersion(D)
        stab = compute_stability(D, k=k_stab)

        res = None
        if gradient_direction is not None:
            res = residual_energy(gradient_direction, K_total, tau=tau, energy=energy)

        profile = None
        if compute_profile:
            profile = explanatory_profile(
                K_total, D,
                gradient=gradient_direction,
                other_layer_grams=other_layer_grams,
                other_layer_distances=other_layer_distances,
                tau_dn=tau,
                tau_is=tau,
                eta_H=eta_H,
                eta_W=eta_W,
                energy=energy,
            )

        return LayerGeometry(
            K_W=K_W, K_A=K_A, K_G=K_G, K_total=K_total, D=D,
            spectrum=spectrum, coords_2d=coords_2d, coords_3d=coords_3d,
            coverage=cov, dispersion=disp, stability=stab,
            residual=res, profile=profile, labels=reps.labels,
        )

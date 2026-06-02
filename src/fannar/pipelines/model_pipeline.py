"""Pipeline de modelo: orquestra a análise de múltiplas camadas."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..types import EnergyKind, LayerGeometry
from .layer_pipeline import LayerPipeline, LayerRepresentations


@dataclass
class ModelGeometry:
    """Geometrias por camada e perfis agregados."""

    layers: dict[str, LayerGeometry] = field(default_factory=dict)

    def profile_matrix(self) -> tuple[list[str], torch.Tensor]:
        """Matriz ``(L, 8)`` de perfis (eixos indisponíveis como 0)."""
        names = list(self.layers.keys())
        rows = []
        for name in names:
            g = self.layers[name]
            if g.profile is not None:
                rows.append(g.profile.as_vector())
            else:
                rows.append(torch.zeros(8))
        return names, torch.stack(rows) if rows else torch.empty(0, 8)


class ModelPipeline:
    """Aplica um (ou um por camada) LayerPipeline a um conjunto de camadas."""

    def __init__(
        self,
        layer_pipeline: LayerPipeline | dict[str, LayerPipeline],
    ) -> None:
        self.layer_pipeline = layer_pipeline

    def _pipe_for(self, name: str) -> LayerPipeline:
        if isinstance(self.layer_pipeline, dict):
            return self.layer_pipeline[name]
        return self.layer_pipeline

    def analyze(
        self,
        reps_by_layer: dict[str, LayerRepresentations],
        tau: float = 0.95,
        energy: EnergyKind = "operator",
        gradient_directions: dict[str, torch.Tensor] | None = None,
        compute_profile: bool = True,
    ) -> ModelGeometry:
        # Primeira passada: K_total de cada camada (para α_Un inter-camada).
        totals: dict[str, torch.Tensor] = {}
        for name, reps in reps_by_layer.items():
            pipe = self._pipe_for(name)
            geo = pipe.analyze(reps, tau=tau, energy=energy, compute_profile=False)
            totals[name] = geo.K_total

        out = ModelGeometry()
        for name, reps in reps_by_layer.items():
            pipe = self._pipe_for(name)
            others = [v for k, v in totals.items() if k != name]
            gdir = None if gradient_directions is None else gradient_directions.get(name)
            geo = pipe.analyze(
                reps, tau=tau, energy=energy,
                gradient_direction=gdir,
                other_layer_grams=others if others else None,
                compute_profile=compute_profile,
            )
            out.layers[name] = geo
        return out

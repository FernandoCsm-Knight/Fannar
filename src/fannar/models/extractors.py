"""Extração de representações de modelos PyTorch.

LIMITAÇÕES DECLARADAS
---------------------
- A extração é genérica e funciona para módulos nomeados padrão (Conv2d,
  Linear, blocos de transformer que retornam um tensor). Arquiteturas que
  retornam estruturas aninhadas ou que fundem operações podem exigir um
  adaptador específico.
- Gradientes exigem ``requires_grad`` no caminho até a camada e um alvo
  escalar para o backward. Em ``model.eval()`` isso funciona desde que os
  parâmetros/entradas componham um grafo diferenciável (não use
  ``torch.no_grad()``).
- O reshape para ``(C, d)`` segue :mod:`pytorch_adapter`; verifique se a
  convenção de "canal" corresponde à sua arquitetura.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from ..errors import ExtractionError
from ..pipelines.layer_pipeline import LayerRepresentations
from .hooks import HookManager
from .pytorch_adapter import (
    activation_to_channel_matrix,
    get_module_by_name,
    weight_to_channel_matrix,
)


class FannarExtractor:
    """Extrai pesos/ativações/gradientes por camada de um modelo PyTorch."""

    VALID_CAPTURE = {"weights", "activations", "gradients"}

    def __init__(
        self,
        model: nn.Module,
        layers: Sequence[str],
        capture: Sequence[str] = ("weights", "activations", "gradients"),
    ) -> None:
        self.model = model
        self.layers = list(layers)
        bad = set(capture) - self.VALID_CAPTURE
        if bad:
            raise ExtractionError(f"capture inválido: {bad}. Use {self.VALID_CAPTURE}.")
        self.capture = list(capture)

    def extract(
        self,
        inputs: torch.Tensor,
        target: int | torch.Tensor | None = None,
        contrastive_class: int | None = None,
    ) -> dict[str, LayerRepresentations]:
        """Roda o modelo e coleta as fontes pedidas por camada.

        ``target`` é a classe alvo (logit) para o backward dos gradientes.
        ``contrastive_class`` permite usar ``logit_target - logit_contrast``
        como escalar para o backward (relevância contrastiva).
        """
        need_grad = "gradients" in self.capture
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        results: dict[str, LayerRepresentations] = {
            name: LayerRepresentations() for name in self.layers
        }

        # pesos: independem do forward
        if "weights" in self.capture:
            for name in self.layers:
                mod = get_module_by_name(self.model, name)
                w = weight_to_channel_matrix(mod)
                results[name].weights = None if w is None else w.detach()

        if "activations" not in self.capture and not need_grad:
            return results

        with HookManager(self.model, self.layers, capture_grad=need_grad) as hm:
            if need_grad:
                inputs = inputs.clone().requires_grad_(True)
                logits = self.model(inputs)
                scalar = self._target_scalar(logits, target, contrastive_class)
                self.model.zero_grad(set_to_none=True)
                scalar.backward()
            else:
                with torch.no_grad():
                    self.model(inputs)

            for name in self.layers:
                if "activations" in self.capture and name in hm.activations:
                    act = hm.activations[name].detach()
                    results[name].activations = activation_to_channel_matrix(act)
                if need_grad and name in hm.gradients:
                    grad = hm.gradients[name]
                    results[name].gradients = activation_to_channel_matrix(grad)

        return results

    @staticmethod
    def _target_scalar(
        logits: torch.Tensor,
        target: int | torch.Tensor | None,
        contrastive_class: int | None,
    ) -> torch.Tensor:
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if target is None:
            target = int(logits.argmax(dim=1)[0].item())
        tgt = logits[:, target].sum()
        if contrastive_class is not None:
            tgt = tgt - logits[:, contrastive_class].sum()
        return tgt

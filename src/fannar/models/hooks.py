"""Gerenciador de hooks de ativação e gradiente, com remoção garantida.

Princípio: o modelo nunca é modificado permanentemente. Todos os hooks são
removidos no ``__exit__`` (ou em :meth:`remove`), mesmo sob exceção.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .pytorch_adapter import get_module_by_name


class HookManager:
    """Registra forward hooks para capturar ativações e seus gradientes.

    Uso::

        with HookManager(model, layers) as hm:
            out = model(x)
            out[:, target].sum().backward()
        acts = hm.activations          # dict nome -> tensor
        grads = hm.gradients           # dict nome -> tensor (se capturado)
    """

    def __init__(self, model: nn.Module, layers: Sequence[str], capture_grad: bool = False) -> None:
        self.model = model
        self.layers = list(layers)
        self.capture_grad = capture_grad
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.activations: dict[str, torch.Tensor] = {}
        self.gradients: dict[str, torch.Tensor] = {}

    def _make_hook(self, name: str):
        def hook(_module, _inp, output):
            out = output[0] if isinstance(output, tuple) else output
            self.activations[name] = out
            if self.capture_grad and isinstance(out, torch.Tensor) and out.requires_grad:
                out.retain_grad()

                def _save_grad(grad, _name=name):
                    self.gradients[_name] = grad.detach()

                out.register_hook(_save_grad)
        return hook

    def register(self) -> HookManager:
        for name in self.layers:
            module = get_module_by_name(self.model, name)
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)
        return self

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> HookManager:
        return self.register()

    def __exit__(self, *exc) -> None:
        self.remove()

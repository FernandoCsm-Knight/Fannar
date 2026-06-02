"""Adaptadores utilitários para modelos PyTorch.

Mantém a fronteira clara entre extração (este pacote) e kernels: aqui só se
produzem tensores ``(C, d)`` por fonte; nenhum kernel é calculado.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..errors import ExtractionError


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """Resolve um submódulo por nome pontilhado (ex.: ``'features.3'``)."""
    mod: nn.Module = model
    for part in name.split("."):
        if not hasattr(mod, part):
            # tenta índice numérico em Sequential/ModuleList
            try:
                mod = mod[int(part)]  # type: ignore[index]
                continue
            except (ValueError, TypeError, IndexError, KeyError) as exc:
                raise ExtractionError(f"Submódulo '{name}' não encontrado em '{part}'.") from exc
        mod = getattr(mod, part)
    return mod


def weight_to_channel_matrix(module: nn.Module) -> torch.Tensor | None:
    """Extrai a matriz de pesos como ``(C, d)`` com ``C`` = canais de saída.

    Conv: ``weight`` ``(C_out, C_in, kh, kw)`` -> ``(C_out, C_in*kh*kw)``.
    Linear: ``weight`` ``(out, in)`` -> ``(out, in)``.
    Retorna ``None`` se o módulo não tiver ``weight``.
    """
    w = getattr(module, "weight", None)
    if w is None:
        return None
    return w.detach().reshape(w.shape[0], -1)


def activation_to_channel_matrix(act: torch.Tensor) -> torch.Tensor:
    """Reorganiza uma ativação em ``(C, d)`` com ``C`` = canais.

    - Conv ``(N, C, H, W)`` -> ``(C, N*H*W)``.
    - Linear ``(N, F)`` -> ``(F, N)`` (canais = features).
    - ``(C, d)`` é mantido como está.
    """
    if act.ndim == 4:  # N, C, H, W
        n, c, h, w = act.shape
        return act.permute(1, 0, 2, 3).reshape(c, n * h * w)
    if act.ndim == 3:  # N, T, C  (ex.: transformer) -> canais = C
        n, t, c = act.shape
        return act.permute(2, 0, 1).reshape(c, n * t)
    if act.ndim == 2:  # N, F -> features como canais
        return act.transpose(0, 1)
    if act.ndim == 1:
        return act.unsqueeze(1)
    return act.reshape(act.shape[0], -1)

"""Conveniências para modelos de linguagem: embeddings por token.

LIMITAÇÃO: assume que a saída capturada da camada tem shape ``(N, T, C)`` ou
``(T, C)``. O mapeamento de tokens segue a ordem do tokenizer fornecido pelo
chamador; nenhum tokenizer específico é assumido aqui.
"""

from __future__ import annotations

import torch


def token_matrix(hidden: torch.Tensor) -> torch.Tensor:
    """Converte estados ocultos em matriz ``(T, C)`` (tokens como objetos).

    - ``(N, T, C)`` -> usa a primeira sequência -> ``(T, C)``.
    - ``(T, C)`` -> mantido.
    """
    if hidden.ndim == 3:
        return hidden[0]
    if hidden.ndim == 2:
        return hidden
    raise ValueError(f"Esperado (N,T,C) ou (T,C), recebido {tuple(hidden.shape)}.")

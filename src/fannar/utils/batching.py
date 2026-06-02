"""Batching/chunking para construção de matrizes de Gram grandes."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch


def chunk_ranges(n: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    """Gera pares ``(início, fim)`` cobrindo ``range(n)`` em blocos."""
    for start in range(0, n, chunk_size):
        yield start, min(start + chunk_size, n)


def blockwise_gram(
    x: torch.Tensor,
    y: torch.Tensor,
    block_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Constrói ``K[i,j] = block_fn(x_i, y_j)`` em blocos para poupar memória.

    ``block_fn`` recebe sub-blocos ``(b1, d)`` e ``(b2, d)`` e retorna ``(b1, b2)``.
    """
    n, m = x.shape[0], y.shape[0]
    out = torch.empty((n, m), dtype=x.dtype, device=x.device)
    for i0, i1 in chunk_ranges(n, chunk_size):
        for j0, j1 in chunk_ranges(m, chunk_size):
            out[i0:i1, j0:j1] = block_fn(x[i0:i1], y[j0:j1])
    return out

"""Produto tensorial de RKHS via produto de Hadamard das matrizes de Gram.

Teorema do tensor de similaridade: o produto tensorial de RKHS é (isomorfo a)
um RKHS cujo kernel é o produto dos kernels componentes. Na forma matricial
(Corolário da forma de Hadamard):

    K_total = K_1 ∘ K_2 ∘ ... ∘ K_d

O produto de Hadamard de matrizes PSD é PSD (teorema de Schur).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

from ..errors import ShapeError
from ..utils.linalg import symmetrize
from ..utils.validation import check_same_shape


def _as_tensor(g: object) -> torch.Tensor:
    if isinstance(g, torch.Tensor):
        return g
    if hasattr(g, "values"):  # GramMatrix
        return g.values
    raise ShapeError(f"Esperado torch.Tensor ou GramMatrix, recebido {type(g)}.")


def hadamard_combine(
    grams: Sequence[torch.Tensor],
    stabilize: bool = True,
    eps: float = 1e-12,  # noqa: ARG001 - mantido por simetria de API
) -> torch.Tensor:
    """Produto de Hadamard direto de uma sequência de Grams ``(n, n)``."""
    tensors = [_as_tensor(g) for g in grams]
    if not tensors:
        raise ShapeError("Lista de Grams vazia.")
    check_same_shape(*tensors)
    out = tensors[0].clone()
    for k in tensors[1:]:
        out = out * k
    if stabilize:
        out = symmetrize(out)
    return out


def hadamard_combine_log(
    grams: Sequence[torch.Tensor],
    eps: float = 1e-12,
    stabilize: bool = True,
) -> torch.Tensor:
    """Produto de Hadamard no domínio log, para muitos fatores.

    Mais estável quando há muitos fatores cujo produto direto causa underflow.

    LIMITAÇÃO: o domínio log só é bem-definido para entradas estritamente
    positivas. Entradas <= 0 fazem ``log`` divergir. Aqui o sinal é tratado
    separadamente (produto dos sinais) e a magnitude via ``exp(Σ log|·|)``,
    mas zeros exatos colapsam a entrada combinada a zero. Para Grams com
    valores negativos, prefira :func:`hadamard_combine`.
    """
    tensors = [_as_tensor(g) for g in grams]
    if not tensors:
        raise ShapeError("Lista de Grams vazia.")
    check_same_shape(*tensors)
    stacked = torch.stack(tensors, dim=0)  # (d, n, n)
    sign = torch.sign(stacked).prod(dim=0)
    log_mag = torch.log(stacked.abs().clamp_min(eps)).sum(dim=0)
    is_zero = (stacked == 0).any(dim=0)
    out = sign * torch.exp(log_mag)
    out = torch.where(is_zero, torch.zeros_like(out), out)
    if stabilize:
        out = symmetrize(out)
    return out


class TensorGram:
    """Combinação tensorial de matrizes de Gram via produto de Hadamard."""

    @staticmethod
    def combine(
        grams: Sequence[torch.Tensor],
        mode: Literal["hadamard"] = "hadamard",
        stabilize: bool = True,
        eps: float = 1e-12,
    ):
        """Combina ``grams`` e retorna um :class:`~fannar.gram.matrix.GramMatrix`."""
        from .matrix import GramMatrix  # import tardio evita ciclo

        if mode != "hadamard":
            raise ValueError(f"mode '{mode}' não suportado.")
        values = hadamard_combine(grams, stabilize=stabilize, eps=eps)
        labels = None
        for g in grams:
            if hasattr(g, "labels") and g.labels is not None:
                labels = g.labels
                break
        return GramMatrix(values=values, labels=labels, axis_type="tensor")

    @staticmethod
    def combine_log_domain(
        grams: Sequence[torch.Tensor],
        stabilize: bool = True,
        eps: float = 1e-12,
    ):
        from .matrix import GramMatrix

        values = hadamard_combine_log(grams, eps=eps, stabilize=stabilize)
        return GramMatrix(values=values, axis_type="tensor")

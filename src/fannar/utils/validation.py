"""Validações estruturais de tensores e matrizes de Gram."""

from __future__ import annotations

import warnings

import torch

from ..config import get_config
from ..errors import NotPSDError, NotSymmetricError, ShapeError


def check_square(K: torch.Tensor, name: str = "K") -> None:
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ShapeError(f"{name} deve ser quadrada 2D, recebido shape {tuple(K.shape)}.")


def check_symmetric(
    K: torch.Tensor, atol: float = 1e-6, name: str = "K", strict: bool | None = None
) -> None:
    """Verifica simetria. Em modo não-strict apenas avisa."""
    check_square(K, name)
    asym = (K - K.T).abs().max().item()
    if asym > atol:
        msg = f"{name} não é simétrica (max |K - Kᵀ| = {asym:.3e} > {atol:.1e})."
        if (get_config().strict if strict is None else strict):
            raise NotSymmetricError(msg)
        warnings.warn(msg, stacklevel=2)


def is_psd(K: torch.Tensor, eps: float | None = None) -> bool:
    """Retorna ``True`` se o menor autovalor for >= -eps."""
    eps = get_config().eps_psd if eps is None else eps
    Ks = 0.5 * (K + K.T)
    try:
        evals = torch.linalg.eigvalsh(Ks)
    except Exception:  # pragma: no cover - fallback numérico
        evals = torch.linalg.eigvals(Ks).real
    return bool(evals.min().item() >= -abs(eps))


def check_psd(
    K: torch.Tensor, eps: float | None = None, name: str = "K", strict: bool | None = None
) -> None:
    """Verifica PSD. Em modo não-strict apenas avisa."""
    eps = get_config().eps_psd if eps is None else eps
    if not is_psd(K, eps):
        Ks = 0.5 * (K + K.T)
        mn = torch.linalg.eigvalsh(Ks).min().item()
        msg = f"{name} não é PSD (autovalor mínimo = {mn:.3e} < -{abs(eps):.1e})."
        if (get_config().strict if strict is None else strict):
            raise NotPSDError(msg)
        warnings.warn(msg, stacklevel=2)


def check_same_shape(*tensors: torch.Tensor) -> None:
    shapes = [tuple(t.shape) for t in tensors]
    if len(set(shapes)) > 1:
        raise ShapeError(f"Shapes incompatíveis: {shapes}.")


def warn_large(n: int, name: str = "operação") -> None:
    limit = get_config().max_gram_n
    if n > limit:
        warnings.warn(
            f"{name}: n={n} excede max_gram_n={limit}. Operações densas O(n^2)/O(n^3) "
            f"podem ser lentas ou estourar memória. Considere chunking ou subamostragem.",
            stacklevel=2,
        )

"""Positive-definite kernels that turn one source of information into a raw Gram matrix.

Every kernel maps a set of objects ``X`` (any array-like whose first axis indexes the
objects; the remaining axes are flattened) to the raw Gram matrix
``K[i, j] = kappa(x_i, x_j)``. The method only requires each kernel to be positive
definite; which kernel to use is part of the *instantiation*, not of the method.

A word on composition. The joint Gram is the Hadamard product of the component Grams.
With linear-type kernels this realises the tensor product of the feature spaces, so the
joint similarity keeps the *multiplicative interaction* between sources. With Gaussian
kernels it does not: the product of two Gaussians is the Gaussian of the summed
exponents, and the composition collapses into an additive combination of the source
metrics. :class:`RBF` is available, but ``Linear``/``Cosine`` are the default for this
reason.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


def as_matrix(x) -> np.ndarray:
    """Objects as rows: ``(n, ...)`` array-like (NumPy, PyTorch, lists) -> ``(n, D)`` float64."""
    if hasattr(x, "detach"):  # torch.Tensor without importing torch
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return x.reshape(len(x), -1)


class Kernel(ABC):
    """Base class. Subclasses implement :meth:`compute` on ``(n, D)`` and ``(m, D)`` arrays."""

    name = "kernel"

    def __call__(self, X, Y=None) -> np.ndarray:
        X = as_matrix(X)
        Y = X if Y is None else as_matrix(Y)
        return self.compute(X, Y)

    @abstractmethod
    def compute(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Raw Gram block ``K[i, j] = kappa(X[i], Y[j])``."""

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({params})"


class Linear(Kernel):
    """``kappa(x, y) = <x, y>``. Combined with the angular transform it becomes the cosine."""

    name = "linear"

    def compute(self, X, Y):
        return X @ Y.T


class Cosine(Kernel):
    """``kappa(x, y) = <x, y> / (|x| |y|)``; zero vectors get similarity 0 (1 with themselves)."""

    name = "cosine"

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def compute(self, X, Y):
        nx = np.linalg.norm(X, axis=1)
        ny = np.linalg.norm(Y, axis=1)
        k = (X @ Y.T) / np.maximum(np.outer(nx, ny), self.eps)
        if X is Y:
            np.fill_diagonal(k, 1.0)
        return k


class RBF(Kernel):
    """Gaussian kernel ``exp(-gamma |x - y|^2)``. ``gamma=None`` uses the median heuristic.

    See the module docstring: under Hadamard composition, Gaussian components add their
    metrics instead of multiplying the similarities.
    """

    name = "rbf"

    def __init__(self, gamma: float | None = None) -> None:
        self.gamma = gamma

    def compute(self, X, Y):
        sx = (X**2).sum(1)[:, None]
        sy = (Y**2).sum(1)[None, :]
        d2 = np.maximum(sx + sy - 2 * X @ Y.T, 0.0)
        gamma = self.gamma
        if gamma is None:
            med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
            gamma = 1.0 / med
        return np.exp(-gamma * d2)


class Polynomial(Kernel):
    """``kappa(x, y) = (gamma <x, y> + coef0) ** degree`` with ``coef0 >= 0``."""

    name = "polynomial"

    def __init__(self, degree: int = 2, coef0: float = 1.0, gamma: float | None = None) -> None:
        if coef0 < 0:
            raise ValueError("coef0 must be >= 0 for the kernel to be positive definite")
        self.degree, self.coef0, self.gamma = degree, coef0, gamma

    def compute(self, X, Y):
        gamma = self.gamma if self.gamma is not None else 1.0 / X.shape[1]
        return (gamma * (X @ Y.T) + self.coef0) ** self.degree


class FunctionKernel(Kernel):
    """Wraps a user function ``fn(X, Y) -> (n, m) array``. Positive definiteness is the user's."""

    name = "function"

    def __init__(self, fn: Callable[[np.ndarray, np.ndarray], np.ndarray], name: str | None = None) -> None:
        self.fn = fn
        if name:
            self.name = name

    def compute(self, X, Y):
        return np.asarray(self.fn(X, Y), dtype=np.float64)


_BY_NAME = {"linear": Linear, "cosine": Cosine, "rbf": RBF, "polynomial": Polynomial}


def as_kernel(obj) -> Kernel:
    """Accepts a :class:`Kernel`, a name (``"linear"``, ``"cosine"``, ``"rbf"``, ``"polynomial"``)
    or a callable ``fn(X, Y)``."""
    if isinstance(obj, Kernel):
        return obj
    if isinstance(obj, str):
        try:
            return _BY_NAME[obj.lower()]()
        except KeyError as exc:
            raise ValueError(f"unknown kernel {obj!r}; choose from {sorted(_BY_NAME)}") from exc
    if callable(obj):
        return FunctionKernel(obj)
    raise TypeError(f"cannot interpret {obj!r} as a kernel")

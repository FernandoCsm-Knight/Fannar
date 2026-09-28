"""Admissible post-kernel transformations.

A transformation ``T`` is admissible when it maps symmetric positive semidefinite (PSD)
matrices to symmetric PSD matrices and is equivariant to re-indexing the objects,
``T(P G P^T) = P T(G) P^T``. Admissible transformations are closed under composition,
which :class:`Transform` exposes as ``>>`` (left to right)::

    component = angular >> shift(1.0)      # T_a, then (G + J) / 2
    final     = center >> trace_normalize  # T_c, then T_tr

All entry-wise and row/column-mean transformations here cost O(n^2); the spectral ones
(:func:`whiten`, :func:`spectral`, :func:`psd_project`) need an eigendecomposition, O(n^3).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class Transform:
    """A named admissible transformation; ``a >> b`` applies ``a`` and then ``b``."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], name: str) -> None:
        self.fn, self.name = fn, name

    def __call__(self, G) -> np.ndarray:
        G = np.asarray(G, dtype=np.float64)
        return self.fn(0.5 * (G + G.T))

    def __rshift__(self, other: "Transform") -> "Transform":
        return Transform(lambda G: other(self(G)), f"{self.name} >> {other.name}")

    def __repr__(self) -> str:
        return f"Transform({self.name})"


def _center(G: np.ndarray) -> np.ndarray:
    # H G H without the O(n^3) products: G_ij - mean_i - mean_j + mean
    r = G.mean(1, keepdims=True)
    return G - r - r.T + G.mean()


def _trace(G: np.ndarray) -> np.ndarray:
    t = np.trace(G)
    if t <= 0:
        raise ValueError("trace normalisation needs a matrix with positive trace")
    return G / t


def _frobenius(G: np.ndarray) -> np.ndarray:
    f = np.linalg.norm(G)
    return G / f if f > 0 else G


def _max_eig(G: np.ndarray) -> np.ndarray:
    top = float(np.linalg.eigvalsh(G)[-1])
    return G / top if top > 0 else G


def _angular_fn(eps: float):
    def fn(G):
        d = np.sqrt(np.maximum(np.diag(G), eps))
        out = G / np.outer(d, d)
        np.fill_diagonal(out, np.where(np.diag(G) > eps, 1.0, 0.0))
        return out
    return fn


def _eigh(G: np.ndarray):
    vals, vecs = np.linalg.eigh(G)
    return vals, vecs


identity = Transform(lambda G: G, "id")
center = Transform(_center, "center")
trace_normalize = Transform(_trace, "trace")
frobenius_normalize = Transform(_frobenius, "frobenius")
max_eig_normalize = Transform(_max_eig, "max_eig")


def angular(eps: float = 1e-12) -> Transform:
    """``T_a(G) = D^{-1/2} G D^{-1/2}``: cosine similarity, unit diagonal (0 for null objects)."""
    return Transform(_angular_fn(eps), "angular")


def shift(c: float = 1.0) -> Transform:
    """``(G + c J) / (1 + c)``. Adding the rank-one PSD matrix ``J`` keeps ``G`` PSD; with
    ``c = 1`` a cosine Gram moves into ``[0, 1]``, which keeps a Hadamard product of several
    components from collapsing to the identity."""
    if c < 0:
        raise ValueError("shift must be >= 0 to stay admissible")
    return Transform(lambda G: (G + c) / (1.0 + c), f"shift({c:g})")


def whiten(tol: float = 1e-10) -> Transform:
    """``T_w``: projector onto the range of ``G`` (all positive eigenvalues set to 1)."""
    def fn(G):
        vals, vecs = _eigh(G)
        keep = vals > tol * max(vals.max(), 1e-300)
        v = vecs[:, keep]
        return v @ v.T
    return Transform(fn, "whiten")


def spectral(f: Callable[[np.ndarray], np.ndarray], name: str = "spectral") -> Transform:
    """``V f(Lambda) V^T`` for a function with ``f(lambda) >= 0`` on ``lambda >= 0``."""
    def fn(G):
        vals, vecs = _eigh(G)
        fv = np.asarray(f(np.maximum(vals, 0.0)), dtype=np.float64)
        if np.any(fv < 0):
            raise ValueError("spectral function must be non-negative to stay admissible")
        return (vecs * fv) @ vecs.T
    return Transform(fn, name)


def psd_project() -> Transform:
    """Nearest PSD matrix in Frobenius norm (negative eigenvalues set to zero)."""
    return spectral(lambda v: v, "psd_project")


def hadamard_power(p: float) -> Transform:
    """Entry-wise power ``G ** p``. PSD is guaranteed only for integer ``p`` (or ``p`` large
    enough); use it followed by :func:`psd_project` for fractional powers."""
    return Transform(lambda G: np.sign(G) * np.abs(G) ** p, f"power({p:g})")


def discarded_mass(G) -> float:
    """Fraction of the spectral mass in negative eigenvalues (what a PSD projection throws away)."""
    vals = np.linalg.eigvalsh(0.5 * (np.asarray(G, float) + np.asarray(G, float).T))
    return float(-vals[vals < 0].sum() / max(np.abs(vals).sum(), 1e-300))

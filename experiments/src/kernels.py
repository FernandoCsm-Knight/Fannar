"""Kernels kappa_l and admissible post-kernel transformations T (gram.tex).

A pipeline is a pair Pi = (kappa, T) with T: S_+^C -> S_+^C permutation-equivariant
(Definition `def:T_admissible`). Component pipelines Pi_l = (kappa_l, T_l) are
combined by the Hadamard product G = G_1 o ... o G_d (eq. `transformed_tensor_gram`),
which realises the similarity tensor at the sample level.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# kappa_l : positive definite kernels on X_l
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# T : admissible post-kernel transformations (Table tab:T_catalog)
# --------------------------------------------------------------------------


def T_id(G: np.ndarray) -> np.ndarray:
    return G


def T_c(G: np.ndarray) -> np.ndarray:
    """Centering H G H: removes a common translation in feature space."""
    n = G.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ G @ H


def T_tr(G: np.ndarray) -> np.ndarray:
    """Trace normalisation: tr(G) = 1, relative spectral energy."""
    t = np.trace(G)
    return G / t if abs(t) > 1e-12 else G


def T_F(G: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(G, "fro")
    return G / n if n > 1e-12 else G


def T_lambda(G: np.ndarray) -> np.ndarray:
    lmax = np.linalg.eigvalsh(G)[-1]
    return G / lmax if lmax > 1e-12 else G


def T_a(G: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Angular normalisation D^-1/2 G D^-1/2: discards individual magnitudes."""
    d = np.sqrt(np.maximum(np.diag(G), eps))
    return G / np.outer(d, d)


def T_w(G: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Whitening/projector V_+ V_+^T onto Im(G): keeps the subspace, drops eigenvalues."""
    vals, vecs = np.linalg.eigh(G)
    keep = vals > tol * max(vals.max(), 1.0)
    Vp = vecs[:, keep]
    return Vp @ Vp.T


def T_tr_c(G: np.ndarray) -> np.ndarray:
    return T_tr(T_c(G))


def T_tr_a(G: np.ndarray) -> np.ndarray:
    return T_tr(T_a(G))


TRANSFORMS = {
    "id": T_id,
    "center": T_c,
    "trace": T_tr,
    "frobenius": T_F,
    "lambda": T_lambda,
    "angular": T_a,
    "whiten": T_w,
    "trace_center": T_tr_c,
    "trace_angular": T_tr_a,
}


def apply_transform(G: np.ndarray, name: str) -> np.ndarray:
    G = TRANSFORMS[name](G)
    return 0.5 * (G + G.T)


# --------------------------------------------------------------------------
# similarity tensor at sample level
# --------------------------------------------------------------------------


def hadamard(mats: list[np.ndarray]) -> np.ndarray:
    """G = G_1 o ... o G_d; PSD by Schur's theorem.

    Note the joint-similarity criterion of Remark `rem:joint_similarity`: with
    normalised components the diagonal stays 1 while a moderate off-diagonal in
    every factor multiplies down towards 0. With d = 3 and typical entries ~0.3
    the product is ~0.03 and G becomes numerically diagonal, which flattens the
    spectrum and makes the truncation at tau vacuous (r -> C). Widening the
    component kernels (see `k_rbf(scale=...)`) keeps the off-diagonals near 1 and
    restores a genuinely low-rank G.
    """
    out = mats[0].copy()
    for M in mats[1:]:
        out *= M
    return out


def project_psd(G: np.ndarray) -> np.ndarray:
    """Clip tiny negative eigenvalues introduced by floating point error."""
    vals, vecs = np.linalg.eigh(0.5 * (G + G.T))
    vals = np.maximum(vals, 0.0)
    return (vecs * vals) @ vecs.T

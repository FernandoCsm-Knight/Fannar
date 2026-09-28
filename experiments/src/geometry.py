"""Ingredientes da Gram por objeto (distance.tex, gram.tex).

  inner_product      <x_n, x_m> / D sobre o eixo achatado (canais x posicoes)
  component_gram     a componente de uma fonte: linear, angular e deslocada para [0, 1]
  gram_distance      a pseudodistancia d(i, j)^2 = G_ii + G_jj − 2 G_ij
  rsa_euclid         a referencia de literatura (RSA com distancia euclidiana)
  mean_class_distance  distancia media de cada objeto a cada classe
"""

from __future__ import annotations


import numpy as np
import torch

from .kernels import apply_transform


# --------------------------------------------------------------------------
# the parameter metric
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Grams over images
# --------------------------------------------------------------------------


def inner_product(x: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    """<x_n, x_m> / D over the flattened (channels x positions) axis."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.from_numpy(np.ascontiguousarray(x)).to(device).float().flatten(1)
    inner = (t @ t.T) / t.shape[1]
    out = inner.double().cpu().numpy()
    del t, inner
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 0.5 * (out + out.T)


def component_gram(
    x: np.ndarray, shift: float = 1.0, device: torch.device | None = None
) -> np.ndarray:
    """Cosine Gram of one source, shifted into [0, 1]. Diagonal is exactly 1.

    Three choices, each forced by a measurement rather than assumed:

    *Linear kernel.* The Hadamard product is supposed to realise the similarity
    tensor, and for linear kernels it does exactly that: the elementwise product
    of two linear Grams is the Gram of the outer products, so the joint really
    lives in the product space. For a Gaussian or Laplacian kernel it does not,
    and not approximately -- the product of two Gaussians is the Gaussian of the
    summed exponents, so the "tensor" composition is *identically* an additive
    combination of the sources' metrics at every bandwidth. That is an algebraic
    identity, not a numerical artefact, and no choice of scale escapes it.

    *Angular normalisation.* Without it the factor with the larger dynamic range
    swallows the other: measured, the joint ranked with the gradient source at
    0.95-1.00 and with the activation source at 0.01-0.07. Angular puts both
    factors on the same footing by construction, diagonal 1 and entries bounded.

    *The shift.* Cosines sit around zero, so multiplying two of them drives the
    off-diagonal to zero -- the joint Gram goes numerically diagonal (median
    off-diagonal 0.003) and the domination returns anyway. Adding a multiple of
    the all-ones matrix is admissible, since it is rank-one positive
    semidefinite, and moving the entries into [0, 1] leaves the product sensitive
    to both factors: median off-diagonal 0.41, and the joint then ranks with the
    activation source at 0.27-0.87 as well as with the gradient source.
    """
    k = apply_transform(inner_product(x, device), "angular")
    return (k + shift) / (1.0 + shift)


def gram_distance(gram: np.ndarray) -> np.ndarray:
    """d(i, j)^2 = G_ii + G_jj - 2 G_ij."""
    diag = np.diag(gram)
    d2 = np.maximum(diag[:, None] + diag[None, :] - 2.0 * gram, 0.0)
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(d2)


# --------------------------------------------------------------------------
# baselines: same objects, distances that do not use the method
# --------------------------------------------------------------------------


def rsa_euclid(activations: np.ndarray) -> np.ndarray:
    """Classical RSA: Euclidean distance between flattened activations.

    Note this is *not* an independent baseline for a Gram built with an RBF or
    Laplacian kernel on the same activations: those kernels are monotone in the
    Euclidean distance, and centring and trace normalisation leave Gram distances
    monotone too, so every rank-based read-out returns identical numbers. It is
    reported because the literature reports it, and the identity itself is worth
    stating rather than presenting the two as if they were competitors.
    """
    x = activations.reshape(len(activations), -1).astype(np.float64)
    sq = (x**2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (x @ x.T), 0.0)
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(d2)


# --------------------------------------------------------------------------
# projection and read-outs
# --------------------------------------------------------------------------


def mean_class_distance(d: np.ndarray, labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """(N, K) mean distance from each sample to each class, excluding itself."""
    out = np.zeros((len(d), n_classes))
    for c in range(n_classes):
        mask = labels == c
        size = int(mask.sum())
        if size == 0:
            out[:, c] = np.inf
            continue
        sums = d[:, mask].sum(1)  # d_ii = 0, so self drops out of the sum
        out[:, c] = sums / np.where(labels == c, max(size - 1, 1), size)
    return out



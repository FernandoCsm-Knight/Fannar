"""Núcleo geométrico: matrizes de Gram e operações sobre o RKHS."""

from .contraction import spectral_contract
from .distance import distance_matrix
from .eigenspace import (
    eigendecompose,
    embedding_fidelity,
    energy_decomposition,
    principal_subspace,
    select_rank,
    spectral_coordinates,
)
from .matrix import GramMatrix
from .psd import clamp_negative_eigenvalues, is_psd, project_psd
from .tensor import (
    TensorGram,
    hadamard_combine,
    hadamard_combine_log,
)

__all__ = [
    "GramMatrix",
    "TensorGram",
    "hadamard_combine",
    "hadamard_combine_log",
    "distance_matrix",
    "eigendecompose",
    "principal_subspace",
    "select_rank",
    "energy_decomposition",
    "spectral_coordinates",
    "embedding_fidelity",
    "spectral_contract",
    "is_psd",
    "project_psd",
    "clamp_negative_eigenvalues",
]

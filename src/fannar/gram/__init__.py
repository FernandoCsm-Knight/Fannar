"""Núcleo geométrico: matrizes de Gram e operações sobre o RKHS."""

from .contraction import spectral_contract
from .distance import (
    affinity_from_distance,
    cumulative_tensor_distances,
    distance_matrix,
    distance_to_ref,
    level_crossing_matrix,
    level_density,
    level_set,
    pairwise_sq_dists,
    sentence_distance_matrix,
)
from .eigenspace import (
    classical_mds,
    eigendecompose,
    embedding_fidelity,
    energy_decomposition,
    kernel_pca,
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
    "pairwise_sq_dists",
    "affinity_from_distance",
    "cumulative_tensor_distances",
    "distance_matrix",
    "sentence_distance_matrix",
    "level_crossing_matrix",
    "distance_to_ref",
    "level_set",
    "level_density",
    "classical_mds",
    "eigendecompose",
    "kernel_pca",
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

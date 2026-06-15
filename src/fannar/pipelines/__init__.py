"""Pipelines de análise: kernel, camada, modelo, explicação."""

from .explain import default_layer_pipeline
from .kernel_pipeline import KernelPipeline
from .layer_pipeline import LayerPipeline, LayerRepresentations
from .model_pipeline import ModelGeometry, ModelPipeline
from .sample_geometry import build_sample_kernel, class_pair_means, pool_sequence
from .token_geometry import cumulative_token_distances, token_distance_matrix

__all__ = [
    "KernelPipeline",
    "LayerPipeline",
    "LayerRepresentations",
    "ModelPipeline",
    "ModelGeometry",
    "default_layer_pipeline",
    "build_sample_kernel",
    "class_pair_means",
    "pool_sequence",
    "token_distance_matrix",
    "cumulative_token_distances",
]

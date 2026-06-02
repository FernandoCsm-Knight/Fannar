"""Pipelines padrão para análise geométrica de camadas.

O pipeline padrão segue a especificação:

    K_layer = K_W ∘ K_A ∘ K_G
    pesos:     LinearKernel + (centramento, normalização por traço)
    ativações: CosineKernel + normalização por traço
    gradientes:LinearKernel + normalização por traço
"""

from __future__ import annotations

from ..kernels.cosine import CosineKernel
from ..kernels.linear import LinearKernel
from ..transforms.centering import CenteringTransform
from ..transforms.composition import ComposeTransforms
from ..transforms.normalization import TraceNormalizeTransform
from .kernel_pipeline import KernelPipeline
from .layer_pipeline import LayerPipeline


def default_layer_pipeline() -> LayerPipeline:
    """Pipeline de camada padrão (K_W ∘ K_A ∘ K_G)."""
    weight_pipe = KernelPipeline(
        LinearKernel(), ComposeTransforms([CenteringTransform(), TraceNormalizeTransform()])
    )
    activation_pipe = KernelPipeline(CosineKernel(), TraceNormalizeTransform())
    gradient_pipe = KernelPipeline(LinearKernel(), TraceNormalizeTransform())
    return LayerPipeline(weights=weight_pipe, activations=activation_pipe, gradients=gradient_pipe)

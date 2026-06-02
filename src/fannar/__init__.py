"""fannar — interpretabilidade geométrica de modelos de IA via RKHS.

Núcleo matemático: kernels -> matrizes de Gram -> transformações pós-kernel ->
produto tensorial (Hadamard) -> decomposição espectral -> energia induzida por
Gram no espaço de representação -> curvas de nível, subconceitos e diagnósticos.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Config
from .concepts import ConceptBank
from .config import FannarConfig, get_config, set_config

# Diagnósticos
from .diagnostics import (
    build_laplacian,
    coverage,
    dispersion,
    explanatory_profile,
    residual_energy,
    stability,
)

# Núcleo
from .gram import (
    GramMatrix,
    TensorGram,
    distance_matrix,
    eigendecompose,
    embedding_fidelity,
    energy_decomposition,
    hadamard_combine,
    principal_subspace,
    spectral_coordinates,
)
from .kernels import (
    CosineKernel,
    LinearKernel,
    PolynomialKernel,
    RBFKernel,
)

# Pipelines / explicação
from .pipelines import (
    KernelPipeline,
    LayerPipeline,
    LayerRepresentations,
    ModelPipeline,
    default_layer_pipeline,
)
from .transforms import (
    AngularNormalizeTransform,
    CenteringTransform,
    ComposeTransforms,
    FrobeniusNormalizeTransform,
    IdentityTransform,
    MaxEigenvalueNormalizeTransform,
    SpectralWhiteningTransform,
    TraceNormalizeTransform,
)

# Tipos
from .types import (
    EnergyDecomposition,
    ExplanatoryProfile,
    LayerGeometry,
    PrincipalSubspace,
    SpectralDecomposition,
)

__all__ = [
    "__version__",
    # config
    "FannarConfig", "get_config", "set_config",
    # kernels
    "LinearKernel", "CosineKernel", "RBFKernel", "PolynomialKernel",
    # transforms
    "IdentityTransform", "CenteringTransform", "TraceNormalizeTransform",
    "FrobeniusNormalizeTransform", "MaxEigenvalueNormalizeTransform",
    "AngularNormalizeTransform", "SpectralWhiteningTransform", "ComposeTransforms",
    # gram
    "GramMatrix", "TensorGram", "hadamard_combine", "distance_matrix",
    "eigendecompose", "principal_subspace", "energy_decomposition",
    "spectral_coordinates", "embedding_fidelity",
    # diagnostics
    "dispersion", "stability", "coverage", "residual_energy",
    "build_laplacian", "explanatory_profile",
    # pipelines
    "KernelPipeline", "LayerPipeline", "LayerRepresentations", "ModelPipeline",
    "default_layer_pipeline",
    # concepts
    "ConceptBank",
    # types
    "SpectralDecomposition", "PrincipalSubspace", "EnergyDecomposition",
    "ExplanatoryProfile", "LayerGeometry",
]

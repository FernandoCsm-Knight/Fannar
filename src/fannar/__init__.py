"""Fannar -- representation spaces of trained models from Gram matrices of several sources.

The method (fixed): one positive-definite kernel per source of information about each object,
admissible transformations, Hadamard composition, and the geometry and spectral energy of the
resulting PSD Gram matrix. The instantiation (your choice): which objects, which sources, which
kernels and transformations.

Quick start::

    import fannar as fa
    interp = fa.Interpreter(model, X_test, y_test, layers=["layer1", "layer2"])
    report = interp.evaluate()
    interp.plot_report(report)
    interp.explain_pair(0).top(5)

By kind of data (thin layers over :class:`Interpreter` with sensible defaults)::

    fa.tabular.TabularInterpreter(model, df_test, y_test)        # PyTorch model on tabular rows
    fa.image.ImageInterpreter(cnn, loader, y_test, layers=[...])  # CNN / ViT, (N, C, H, W) outputs
    fa.text.TextInterpreter(fa.text.wrap_hf(hf_model), ids, y)    # transformers over tokens
    fa.tree.TreeInterpreter(tree, df_test, y_test, X_train=df_train)

Building blocks::

    pipe  = fa.KernelPipeline({"A": fa.Component("linear"), "B": fa.Component(my_kernel)})
    space = pipe(sources={"A": feats_a, "B": feats_b})     # or pipe(grams={"A": K_a, "B": K_b})
    space.distances, space.neighbors(0, 5), space.energies(), space.embedding()
"""

from . import kernels, readings, transforms
from .interpreter import Interpreter
from .kernels import RBF, Cosine, FunctionKernel, Kernel, Linear, Polynomial
from .pairs import ErrorExplanation, PairExplanation
from .pipeline import (PRESETS, Component, KernelPipeline, paper_pipeline, positions_pipeline,
                       single_source)
from .readings import evaluate
from .report import Report
from .sources import FunctionSources, Sources, TorchSources, TreeSources
from .space import GramSpace
from .transforms import (Transform, angular, center, frobenius_normalize, identity, max_eig_normalize,
                         psd_project, shift, spectral, trace_normalize, whiten)

# A versão vem do pyproject.toml (metadados do pacote instalado): um só lugar para mudar.
try:
    from importlib.metadata import PackageNotFoundError as _NotFound, version as _version
    __version__ = _version("fannar")
except _NotFound:  # rodando do código-fonte sem instalar
    __version__ = "0.4.0"

# Módulos por tipo de dado, carregados sob demanda: `import fannar` continua leve e sem torch.
_DOMAINS = ("tabular", "image", "text", "tree")


def __getattr__(name):
    if name in _DOMAINS:
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_DOMAINS))

__all__ = [
    "Interpreter", "GramSpace", "KernelPipeline", "Component", "Report", "PairExplanation", "ErrorExplanation",
    "Sources", "TorchSources", "TreeSources", "FunctionSources",
    "Kernel", "Linear", "Cosine", "RBF", "Polynomial", "FunctionKernel",
    "Transform", "identity", "center", "trace_normalize", "frobenius_normalize", "max_eig_normalize",
    "angular", "shift", "whiten", "spectral", "psd_project",
    "paper_pipeline", "single_source", "positions_pipeline", "PRESETS", "evaluate",
    "kernels", "transforms", "readings", "__version__",
    "tabular", "image", "text", "tree",
]

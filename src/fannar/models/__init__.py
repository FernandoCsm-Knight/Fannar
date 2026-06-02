"""Extração de representações de modelos PyTorch."""

from .extractors import FannarExtractor
from .hooks import HookManager
from .language import token_matrix
from .pytorch_adapter import (
    activation_to_channel_matrix,
    get_module_by_name,
    weight_to_channel_matrix,
)
from .vision import patch_grid_labels, patchify

__all__ = [
    "FannarExtractor",
    "HookManager",
    "get_module_by_name",
    "weight_to_channel_matrix",
    "activation_to_channel_matrix",
    "patchify",
    "patch_grid_labels",
    "token_matrix",
]

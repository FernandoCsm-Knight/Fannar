"""Fixtures comuns dos testes."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(1234)


@pytest.fixture
def X():
    """12 objetos, 8 features, float64."""
    return torch.randn(12, 8, dtype=torch.float64)


@pytest.fixture
def grams(X):
    import fannar as f
    K1 = f.LinearKernel()(X)
    K2 = f.CosineKernel()(X)
    K3 = f.LinearKernel()(torch.randn(12, 8, dtype=torch.float64))
    return K1, K2, K3

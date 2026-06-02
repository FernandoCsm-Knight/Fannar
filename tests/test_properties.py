"""Testes de propriedade. Usa hypothesis se disponível; caso contrário, pula."""

from __future__ import annotations

import pytest
import torch

import fannar as f
from fannar.gram import is_psd

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


@settings(max_examples=25, deadline=None)
@given(n=st.integers(min_value=2, max_value=20), d=st.integers(min_value=1, max_value=10))
def test_linear_always_psd(n, d):
    X = torch.randn(n, d, dtype=torch.float64)
    K = f.LinearKernel()(X)
    assert is_psd(K, eps=1e-6)


@settings(max_examples=25, deadline=None)
@given(n=st.integers(min_value=2, max_value=15))
def test_hadamard_preserves_psd_property(n):
    X1 = torch.randn(n, 5, dtype=torch.float64)
    X2 = torch.randn(n, 5, dtype=torch.float64)
    K = f.hadamard_combine([f.LinearKernel()(X1), f.CosineKernel()(X2)])
    assert is_psd(K, eps=1e-6)


@settings(max_examples=25, deadline=None)
@given(n=st.integers(min_value=3, max_value=15))
def test_residual_ratio_in_unit_interval(n):
    X = torch.randn(n, 6, dtype=torch.float64)
    K = f.LinearKernel()(X)
    sub = f.principal_subspace(K, tau=0.8)
    u = torch.randn(n, dtype=torch.float64)
    ed = f.energy_decomposition(u, sub)
    r = float(ed.residual_ratio)
    assert -1e-9 <= r <= 1.0 + 1e-9

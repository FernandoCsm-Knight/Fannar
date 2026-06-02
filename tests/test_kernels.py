"""Testes de kernels."""

from __future__ import annotations

import torch

import fannar as f
from fannar.gram import is_psd


def test_kernel_shapes(X):
    K = f.LinearKernel()(X)
    assert K.shape == (12, 12)
    Y = torch.randn(5, 8, dtype=torch.float64)
    K2 = f.LinearKernel()(X, Y)
    assert K2.shape == (12, 5)


def test_linear_kernel_psd(X):
    K = f.LinearKernel()(X)
    assert torch.allclose(K, K.T)
    assert is_psd(K)


def test_cosine_diag_unit(X):
    K = f.CosineKernel()(X)
    assert torch.allclose(torch.diagonal(K), torch.ones(12, dtype=torch.float64), atol=1e-6)


def test_rbf_in_unit_interval(X):
    K = f.RBFKernel()(X)
    assert float(K.min()) >= 0.0
    assert float(K.max()) <= 1.0 + 1e-9
    assert torch.allclose(torch.diagonal(K), torch.ones(12, dtype=torch.float64), atol=1e-6)


def test_polynomial_psd(X):
    K = f.PolynomialKernel(degree=2, coef0=1.0)(X)
    assert is_psd(K, eps=1e-6)

"""Testes de transformações pós-kernel."""

from __future__ import annotations

import torch

import fannar as f
from fannar.gram import is_psd


def test_centering_preserves_symmetry(X):
    K = f.LinearKernel()(X)
    Kc = f.CenteringTransform()(K)
    assert torch.allclose(Kc, Kc.T, atol=1e-9)
    assert is_psd(Kc, eps=1e-6)


def test_trace_normalize_unit_trace(X):
    K = f.LinearKernel()(X)
    Kt = f.TraceNormalizeTransform()(K)
    assert abs(float(torch.diagonal(Kt).sum()) - 1.0) < 1e-9


def test_frobenius_normalize_unit_norm(X):
    K = f.LinearKernel()(X)
    Kf = f.FrobeniusNormalizeTransform()(K)
    assert abs(float(torch.linalg.matrix_norm(Kf, ord="fro")) - 1.0) < 1e-9


def test_angular_diag_unit(X):
    K = f.LinearKernel()(X)
    Ka = f.AngularNormalizeTransform()(K)
    assert torch.allclose(torch.diagonal(Ka), torch.ones(12, dtype=torch.float64), atol=1e-6)


def test_compose(X):
    K = f.LinearKernel()(X)
    T = f.ComposeTransforms([f.CenteringTransform(), f.TraceNormalizeTransform()])
    Kt = T(K)
    assert abs(float(torch.diagonal(Kt).sum()) - 1.0) < 1e-9


def test_whitening_psd(X):
    K = f.LinearKernel()(X)
    Kw = f.SpectralWhiteningTransform(power=-0.5)(K)
    assert torch.allclose(Kw, Kw.T, atol=1e-6)

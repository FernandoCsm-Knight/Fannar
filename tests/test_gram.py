"""Testes do núcleo geométrico (gram)."""

from __future__ import annotations

import torch

import fannar as f
from fannar.gram import is_psd


def test_hadamard_preserves_psd(grams):
    K = f.hadamard_combine(list(grams))
    assert is_psd(K, eps=1e-6)
    assert torch.allclose(K, K.T, atol=1e-9)


def test_hadamard_matches_elementwise(grams):
    K1, K2, K3 = grams
    K = f.hadamard_combine([K1, K2, K3])
    expected = 0.5 * ((K1 * K2 * K3) + (K1 * K2 * K3).T)
    assert torch.allclose(K, expected, atol=1e-9)


def test_distance_diag_zero(grams):
    K = f.hadamard_combine(list(grams))
    D = f.distance_matrix(K)
    assert float(torch.diagonal(D).abs().max()) < 1e-9
    assert torch.allclose(D, D.T, atol=1e-9)


def test_eigendecomposition_reconstructs(grams):
    K = f.hadamard_combine(list(grams))
    dec = f.eigendecompose(K)
    assert torch.allclose(dec.reconstruct(), K, atol=1e-6)
    # ordem decrescente
    assert torch.all(dec.eigenvalues[:-1] >= dec.eigenvalues[1:] - 1e-12)


def test_spectral_coordinates_shape(grams):
    K = f.hadamard_combine(list(grams))
    assert f.spectral_coordinates(K, 2).shape == (12, 2)
    assert f.spectral_coordinates(K, 3).shape == (12, 3)


def test_energy_residual_in_unit_interval(grams):
    K = f.hadamard_combine(list(grams))
    sub = f.principal_subspace(K, tau=0.9)
    u = torch.randn(12, dtype=torch.float64)
    ed = f.energy_decomposition(u, sub)
    assert 0.0 <= float(ed.residual_ratio) <= 1.0


def test_parallel_plus_orthogonal_reconstructs_u(grams):
    K = f.hadamard_combine(list(grams))
    sub = f.principal_subspace(K, tau=0.9)
    u = torch.randn(12, dtype=torch.float64)
    ed = f.energy_decomposition(u, sub)
    assert torch.allclose(ed.parallel + ed.orthogonal, u, atol=1e-8)


def test_full_subspace_zero_residual(grams):
    K = f.hadamard_combine(list(grams))
    sub = f.principal_subspace(K, tau=1.0)  # retém tudo
    u = torch.randn(12, dtype=torch.float64)
    ed = f.energy_decomposition(u, sub)
    assert float(ed.residual_ratio) < 1e-9


def test_nearest_psd_makes_psd():
    A = torch.randn(10, 10, dtype=torch.float64)
    A = A + A.T  # simétrica indefinida
    gm = f.GramMatrix(A).nearest_psd()
    assert is_psd(gm.values, eps=1e-8)

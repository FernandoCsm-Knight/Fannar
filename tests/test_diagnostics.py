"""Testes de diagnósticos."""

from __future__ import annotations

import torch

import fannar as f
from fannar.types import ExplanatoryProfile


def test_coverage_bounds(grams):
    K = f.hadamard_combine(list(grams))
    cov = f.coverage(K, tau=0.95)
    assert 0.0 < float(cov.value) <= 1.0
    assert cov.detail["captured_fraction"] >= 0.95 - 1e-6


def test_dispersion_nonnegative(grams):
    K = f.hadamard_combine(list(grams))
    D = f.distance_matrix(K)
    assert float(f.dispersion(D).value) >= 0.0


def test_stability_per_item(grams):
    K = f.hadamard_combine(list(grams))
    D = f.distance_matrix(K)
    stab = f.stability(D, k=3)
    assert stab.detail["per_item"].shape == (12,)


def test_profile_axes_in_unit_interval(grams):
    K = f.hadamard_combine(list(grams))
    D = f.distance_matrix(K)
    g = torch.rand(12, dtype=torch.float64)
    prof = f.explanatory_profile(K, D, gradient=g)
    for ax in ExplanatoryProfile.AXES:
        res = getattr(prof, ax)
        if res.available and res.value is not None:
            assert 0.0 <= float(res.value) <= 1.0


def test_profile_unavailable_without_gradient(grams):
    K = f.hadamard_combine(list(grams))
    prof = f.explanatory_profile(K)
    assert prof.IS.available is False
    assert prof.RE.available is False
    assert prof.Pr.available is False  # sem rótulos
    assert prof.Un.available is False  # sem outras camadas


def test_laplacian_eigenvalues_nonneg(grams):
    K = f.hadamard_combine(list(grams))
    D = f.distance_matrix(K)
    lap = f.build_laplacian(D)
    assert float(lap.eigenvalues.min()) >= -1e-8

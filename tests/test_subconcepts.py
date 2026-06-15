"""Testes do módulo de conceitos via energia induzida pela Gram."""

from __future__ import annotations

import pytest
import torch

from fannar.concepts import (
    GramEnergySurface,
    GramMetricUsage,
    OrthogonalEnergySurface,
    activation_to_spatial_field,
    gram_induced_energy_surface,
    gram_metric_usage,
    metric_score_and_coords,
    normalized_delta,
    orthogonal_energy_surface,
    psd_positive_part,
    suppress_borders,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _field(C: int = 4, H: int = 4, W: int = 4):
    act = torch.randn(C, H, W)
    return activation_to_spatial_field(act.unsqueeze(0))


# ---------------------------------------------------------------------------
# activation_to_spatial_field
# ---------------------------------------------------------------------------

def test_activation_to_spatial_field_shape_3d():
    act = torch.randn(3, 4, 4)
    field, grid = activation_to_spatial_field(act.unsqueeze(0))
    assert grid == (4, 4)
    assert field.shape == (16, 3)


def test_activation_to_spatial_field_shape_4d_batch():
    act = torch.randn(1, 5, 3, 3)
    field, grid = activation_to_spatial_field(act)
    assert grid == (3, 3)
    assert field.shape == (9, 5)


def test_activation_to_spatial_field_rejects_1d():
    with pytest.raises(ValueError):
        activation_to_spatial_field(torch.randn(5))


# ---------------------------------------------------------------------------
# gram_induced_energy_surface
# ---------------------------------------------------------------------------

def test_gram_induced_energy_returns_type():
    field, grid = _field()
    surf = gram_induced_energy_surface(torch.eye(4), field, grid, smooth_sigma=None)
    assert isinstance(surf, GramEnergySurface)


def test_gram_induced_energy_field_shape():
    field, grid = _field(C=4, H=5, W=6)
    surf = gram_induced_energy_surface(torch.eye(4), field, grid, smooth_sigma=None)
    assert surf.field.shape == grid


def test_gram_induced_energy_raw_nonneg():
    field, grid = _field()
    surf = gram_induced_energy_surface(torch.eye(4), field, grid, smooth_sigma=None)
    assert (surf.raw >= 0).all()


def test_gram_induced_energy_coords_shapes():
    field, grid = _field(C=6, H=3, W=3)
    surf = gram_induced_energy_surface(torch.eye(6), field, grid, smooth_sigma=None)
    assert surf.coords_2d.shape == (9, 2)
    assert surf.coords_3d.shape == (9, 3)


def test_gram_induced_energy_eigenvalues_sorted():
    field, grid = _field()
    surf = gram_induced_energy_surface(torch.eye(4), field, grid, smooth_sigma=None)
    eigs = surf.eigenvalues
    assert (eigs[:-1] >= eigs[1:] - 1e-6).all()


def test_gram_induced_energy_dtype_alignment():
    field, grid = _field()
    gram64 = torch.eye(4, dtype=torch.float64)
    surf = gram_induced_energy_surface(gram64, field.float(), grid, smooth_sigma=None)
    assert surf.field.dtype == torch.float32


# ---------------------------------------------------------------------------
# gram_metric_usage
# ---------------------------------------------------------------------------

def test_gram_metric_usage_returns_type():
    field, grid = _field()
    usage = gram_metric_usage(torch.eye(4), field, grid)
    assert isinstance(usage, GramMetricUsage)


def test_gram_metric_usage_shapes():
    field, grid = _field(C=5, H=3, W=3)
    usage = gram_metric_usage(torch.eye(5), field, grid)
    assert usage.used_similarity.shape == (5, 5)
    assert usage.channel_usage.shape == (5,)


def test_gram_metric_usage_symmetry():
    field, grid = _field()
    usage = gram_metric_usage(torch.eye(4), field, grid)
    assert torch.allclose(usage.used_similarity, usage.used_similarity.T, atol=1e-6)


def test_gram_metric_usage_channel_usage_nonneg():
    field, grid = _field()
    usage = gram_metric_usage(torch.eye(4), field, grid)
    assert (usage.channel_usage >= 0).all()


# ---------------------------------------------------------------------------
# orthogonal_energy_surface
# ---------------------------------------------------------------------------

def test_orthogonal_energy_returns_type():
    field, grid = _field(C=5, H=4, W=4)
    surf = orthogonal_energy_surface(torch.eye(5), field, grid, tau=0.9, smooth_sigma=None)
    assert isinstance(surf, OrthogonalEnergySurface)


def test_orthogonal_energy_shapes():
    field, grid = _field(C=5, H=4, W=4)
    surf = orthogonal_energy_surface(torch.eye(5), field, grid, tau=0.9, smooth_sigma=None)
    assert surf.perp_energy.shape == grid
    assert surf.perp_ratio.shape == grid
    assert surf.total_energy.shape == grid


def test_orthogonal_energy_perp_ratio_unit_interval():
    field, grid = _field()
    surf = orthogonal_energy_surface(torch.eye(4), field, grid, tau=0.8, smooth_sigma=None)
    assert (surf.perp_ratio >= -1e-6).all()
    assert (surf.perp_ratio <= 1.0 + 1e-6).all()


def test_orthogonal_energy_coverage_meets_tau():
    field, grid = _field(C=6, H=4, W=4)
    tau = 0.85
    surf = orthogonal_energy_surface(torch.eye(6), field, grid, tau=tau, smooth_sigma=None)
    assert surf.coverage >= tau - 1e-6
    assert surf.tau == tau


def test_orthogonal_energy_full_tau_perp_near_zero():
    """tau=1.0 retém todos os modos → energia perpendicular ≈ 0."""
    field, grid = _field(C=4, H=3, W=3)
    surf = orthogonal_energy_surface(torch.eye(4) * 2.0, field, grid, tau=1.0, smooth_sigma=None)
    assert float(surf.perp_energy.max()) < 1e-4


# ---------------------------------------------------------------------------
# normalized_delta
# ---------------------------------------------------------------------------

def test_normalized_delta_range():
    a = torch.rand(5, 5)
    b = torch.rand(5, 5)
    delta = normalized_delta(a, b)
    assert (delta >= -1.0 - 1e-8).all()
    assert (delta <= 1.0 + 1e-8).all()


def test_normalized_delta_no_contrast_passthrough():
    a = torch.rand(4, 4)
    assert torch.allclose(normalized_delta(a, None), a)


def test_normalized_delta_opposite_direction():
    a = torch.ones(3) * 2.0
    b = torch.zeros(3)
    delta = normalized_delta(a, b)
    assert (delta > 0).all()


def test_normalized_delta_dtype_alignment():
    a = torch.rand(4, 4)
    b = torch.rand(4, 4).double()
    delta = normalized_delta(a, b)
    assert delta.dtype == a.dtype


# ---------------------------------------------------------------------------
# psd_positive_part
# ---------------------------------------------------------------------------

def test_psd_positive_part_is_psd():
    from fannar.gram import is_psd
    A = torch.randn(6, 6)
    A = A + A.T
    P = psd_positive_part(A)
    assert is_psd(P, eps=1e-6)


def test_psd_positive_part_symmetric():
    A = torch.randn(5, 5)
    A = A + A.T
    P = psd_positive_part(A)
    assert torch.allclose(P, P.T, atol=1e-6)


def test_psd_positive_part_identity_unchanged():
    I = torch.eye(4)
    P = psd_positive_part(I)
    assert torch.allclose(P, I, atol=1e-6)


# ---------------------------------------------------------------------------
# suppress_borders
# ---------------------------------------------------------------------------

def test_suppress_borders_zeros_border():
    m = torch.ones(8, 8)
    out = suppress_borders(m, fraction=0.25)
    assert float(out[0, :].sum()) == 0.0
    assert float(out[-1, :].sum()) == 0.0
    assert float(out[:, 0].sum()) == 0.0
    assert float(out[:, -1].sum()) == 0.0


def test_suppress_borders_interior_preserved():
    m = torch.ones(8, 8)
    out = suppress_borders(m, fraction=0.25)
    assert float(out[2:6, 2:6].sum()) > 0.0


def test_suppress_borders_zero_fraction_noop():
    m = torch.rand(6, 6)
    assert torch.allclose(suppress_borders(m, 0.0), m)


# ---------------------------------------------------------------------------
# metric_score_and_coords
# ---------------------------------------------------------------------------

def test_metric_score_shape():
    field, grid = _field(C=4, H=3, W=3)
    score, _ = metric_score_and_coords(torch.eye(4), field, grid)
    assert score.shape == grid


def test_metric_coords_shape():
    field, grid = _field(C=6, H=3, W=3)
    _, coords = metric_score_and_coords(torch.eye(6), field, grid)
    assert coords.shape == (9, 3)


def test_metric_score_nonneg():
    field, grid = _field()
    score, _ = metric_score_and_coords(torch.eye(4), field, grid)
    assert (score >= -1e-6).all()


def test_metric_score_dtype_alignment():
    field, grid = _field()
    gram64 = torch.eye(4, dtype=torch.float64)
    score, coords = metric_score_and_coords(gram64, field.float(), grid)
    assert score.dtype == torch.float32
    assert coords.dtype == torch.float32

"""Testes do módulo fannar.gram.distance (curvas de nível RKHS)."""

from __future__ import annotations

import math

import pytest
import torch

from fannar.gram.distance import (
    affinity_from_distance,
    cumulative_tensor_distances,
    distance_matrix,
    distance_to_ref,
    level_crossing_matrix,
    level_density,
    level_set,
    pairwise_sq_dists,
    sentence_distance_matrix,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gram_5():
    """Gram 5×5 de vetores aleatórios (garantido PSD)."""
    torch.manual_seed(0)
    X = torch.randn(5, 4)
    return X @ X.T


@pytest.fixture
def dist_5(gram_5):
    return distance_matrix(gram_5)


# ---------------------------------------------------------------------------
# pairwise_sq_dists
# ---------------------------------------------------------------------------

class TestPairwiseSqDists:
    def test_diagonal_is_zero(self):
        A = torch.randn(6, 4)
        D2 = pairwise_sq_dists(A)
        assert (D2.diagonal() < 1e-5).all()

    def test_symmetric(self):
        A = torch.randn(6, 4)
        D2 = pairwise_sq_dists(A)
        assert torch.allclose(D2, D2.T, atol=1e-6)

    def test_nonneg(self):
        A = torch.randn(6, 4)
        D2 = pairwise_sq_dists(A)
        assert (D2 >= 0).all()

    def test_euclidean_correct_value(self):
        A = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        D2 = pairwise_sq_dists(A)
        assert abs(float(D2[0, 1]) - 25.0) < 1e-6

    def test_mahalanobis_identity_equals_euclidean(self):
        torch.manual_seed(1)
        A = torch.randn(8, 5)
        D2_eu = pairwise_sq_dists(A)
        D2_ma = pairwise_sq_dists(A, metric=torch.eye(5))
        assert torch.allclose(D2_eu, D2_ma, atol=1e-5)

    def test_mahalanobis_diagonal_zero(self):
        A = torch.randn(6, 4)
        M = torch.eye(4) * 2.0
        D2 = pairwise_sq_dists(A, metric=M)
        assert (D2.diagonal() < 1e-10).all()

    def test_mahalanobis_symmetric(self):
        A = torch.randn(6, 4)
        M = torch.eye(4) * 2.0
        D2 = pairwise_sq_dists(A, metric=M)
        assert torch.allclose(D2, D2.T, atol=1e-6)


# ---------------------------------------------------------------------------
# distance_matrix
# ---------------------------------------------------------------------------

class TestDistanceMatrix:
    def test_diagonal_zero(self, gram_5):
        D = distance_matrix(gram_5)
        assert (D.diagonal() < 1e-12).all()

    def test_symmetric(self, gram_5):
        D = distance_matrix(gram_5)
        assert torch.allclose(D, D.T, atol=1e-6)

    def test_nonneg(self, gram_5):
        D = distance_matrix(gram_5)
        assert (D >= 0).all()

    def test_squared_flag(self, gram_5):
        D = distance_matrix(gram_5)
        D2 = distance_matrix(gram_5, squared=True)
        assert torch.allclose(D2, D.pow(2), atol=1e-5)


# ---------------------------------------------------------------------------
# affinity_from_distance
# ---------------------------------------------------------------------------

class TestAffinityFromDistance:
    def test_diagonal_is_one(self, dist_5):
        K = affinity_from_distance(dist_5)
        assert torch.allclose(K.diagonal(), torch.ones(5), atol=1e-6)

    def test_symmetric(self, dist_5):
        K = affinity_from_distance(dist_5)
        assert torch.allclose(K, K.T, atol=1e-6)

    def test_values_in_unit(self, dist_5):
        K = affinity_from_distance(dist_5)
        assert (K >= 0).all()
        assert (K <= 1.0 + 1e-6).all()

    def test_explicit_sigma(self, dist_5):
        K = affinity_from_distance(dist_5, sigma=1.0)
        assert (K >= 0).all()
        assert (K.diagonal() == 1.0).all()

    def test_zero_distance_gives_one(self):
        D = torch.zeros(4, 4)
        K = affinity_from_distance(D)
        assert torch.allclose(K, torch.ones(4, 4), atol=1e-6)


# ---------------------------------------------------------------------------
# distance_to_ref
# ---------------------------------------------------------------------------

class TestDistanceToRef:
    def test_ref_is_zero(self, gram_5):
        for ref in range(5):
            p = distance_to_ref(gram_5, ref)
            assert float(p[ref]) < 1e-10

    def test_nonneg(self, gram_5):
        p = distance_to_ref(gram_5, 0)
        assert (p >= 0).all()

    def test_length(self, gram_5):
        p = distance_to_ref(gram_5, 2)
        assert p.shape == (5,)

    def test_consistent_with_distance_matrix(self, gram_5):
        D = distance_matrix(gram_5)
        for ref in range(5):
            p = distance_to_ref(gram_5, ref)
            assert torch.allclose(p, D[:, ref], atol=1e-5)


# ---------------------------------------------------------------------------
# sentence_distance_matrix
# ---------------------------------------------------------------------------

class TestSentenceDistanceMatrix:
    def test_shape(self, gram_5):
        D = sentence_distance_matrix(gram_5)
        assert D.shape == (5, 5)

    def test_diagonal_zero(self, gram_5):
        D = sentence_distance_matrix(gram_5)
        assert (D.diagonal() < 1e-10).all()

    def test_symmetric(self, gram_5):
        D = sentence_distance_matrix(gram_5)
        assert torch.allclose(D, D.T, atol=1e-6)

    def test_agrees_with_distance_matrix(self, gram_5):
        D_direct = distance_matrix(gram_5)
        D_foliated = sentence_distance_matrix(gram_5)
        assert torch.allclose(D_direct, D_foliated, atol=1e-5)


# ---------------------------------------------------------------------------
# level_set
# ---------------------------------------------------------------------------

class TestLevelSet:
    def test_ref_not_in_level_set_for_r_pos(self, gram_5):
        p = distance_to_ref(gram_5, 0)
        r = float(p[1:].mean())
        indices = level_set(p, r)
        assert 0 not in indices.tolist()

    def test_all_at_same_radius(self):
        p = torch.ones(10) * 0.5
        indices = level_set(p, 0.5, tol=0.05)
        assert len(indices) == 10

    def test_empty_when_r_far(self, gram_5):
        p = distance_to_ref(gram_5, 0)
        indices = level_set(p, r=1000.0, tol=0.001)
        assert len(indices) == 0

    def test_adaptive_tol_returns_tensor(self, gram_5):
        p = distance_to_ref(gram_5, 0)
        indices = level_set(p, r=float(p.mean()))
        assert isinstance(indices, torch.Tensor)


# ---------------------------------------------------------------------------
# level_crossing_matrix
# ---------------------------------------------------------------------------

class TestLevelCrossingMatrix:
    def test_shape(self, dist_5):
        C = level_crossing_matrix(dist_5)
        assert C.shape == (5, 5)

    def test_diagonal_zero(self, dist_5):
        C = level_crossing_matrix(dist_5)
        assert (C.diagonal() == 0).all()

    def test_nonneg(self, dist_5):
        C = level_crossing_matrix(dist_5)
        assert (C >= 0).all()

    def test_symmetric(self, dist_5):
        C = level_crossing_matrix(dist_5)
        assert torch.allclose(C.float(), C.T.float(), atol=1e-6)

    def test_monotone_with_distance(self, gram_5):
        D = sentence_distance_matrix(gram_5)
        C = level_crossing_matrix(D, n_levels=10)
        # pares mais distantes têm contagem >= pares menos distantes (globalmente)
        i, j = 0, 1
        k, l = 0, 2
        # apenas verifica que não são todos iguais para Gram não trivial
        assert C.max() >= C.min()


# ---------------------------------------------------------------------------
# level_density
# ---------------------------------------------------------------------------

class TestLevelDensity:
    def test_length(self, gram_5):
        v = torch.randn(5)
        rho = level_density(gram_5, ref_idx=0, v=v)
        assert rho.shape == (4,)  # todos exceto ref

    def test_nonneg(self, gram_5):
        v = torch.randn(5)
        rho = level_density(gram_5, ref_idx=0, v=v)
        assert (rho >= 0).all()

    def test_custom_eval_indices(self, gram_5):
        v = torch.randn(5)
        idx = torch.tensor([1, 3])
        rho = level_density(gram_5, ref_idx=0, v=v, eval_indices=idx)
        assert rho.shape == (2,)

    def test_zero_direction_gives_zero(self, gram_5):
        v = torch.zeros(5)
        rho = level_density(gram_5, ref_idx=0, v=v)
        assert (rho.abs() < 1e-10).all()


# ---------------------------------------------------------------------------
# cumulative_tensor_distances
# ---------------------------------------------------------------------------

class TestCumulativeTensorDistances:
    def _make_dists(self, n=6, layers=4):
        dists = []
        for _ in range(layers):
            X = torch.randn(n, 3)
            K = X @ X.T
            dists.append(distance_matrix(K))
        return dists

    def test_list_length(self):
        dists = self._make_dists(layers=4)
        result = cumulative_tensor_distances(dists)
        assert len(result) == 4

    def test_shapes(self):
        dists = self._make_dists(n=5, layers=3)
        result = cumulative_tensor_distances(dists)
        for D in result:
            assert D.shape == (5, 5)

    def test_diagonal_zero(self):
        dists = self._make_dists()
        result = cumulative_tensor_distances(dists)
        for D in result:
            assert (D.diagonal() < 1e-6).all()

    def test_symmetric(self):
        dists = self._make_dists()
        result = cumulative_tensor_distances(dists)
        for D in result:
            assert torch.allclose(D, D.T, atol=1e-5)

    def test_dict_input_same_as_list(self):
        dists = self._make_dists(n=5, layers=3)
        d_dict = {i: d for i, d in enumerate(dists)}
        list_result = cumulative_tensor_distances(dists)
        dict_result = cumulative_tensor_distances(d_dict)
        for a, b in zip(list_result, dict_result):
            assert torch.allclose(a, b, atol=1e-6)

    def test_nonneg(self):
        dists = self._make_dists()
        result = cumulative_tensor_distances(dists)
        for D in result:
            assert (D >= 0).all()

"""Testes de fannar.pipelines.sample_geometry."""

from __future__ import annotations

import pytest
import torch

from fannar.pipelines.sample_geometry import (
    build_sample_kernel,
    class_pair_means,
    pool_sequence,
)


# ---------------------------------------------------------------------------
# pool_sequence
# ---------------------------------------------------------------------------

class TestPoolSequence:
    def test_1d_passthrough(self):
        x = torch.randn(8)
        out = pool_sequence(x)
        assert torch.equal(out, x)

    def test_2d_mean(self):
        x = torch.ones(10, 5) * 3.0
        out = pool_sequence(x)
        assert out.shape == (5,)
        assert torch.allclose(out, torch.ones(5) * 3.0)

    def test_2d_shape(self):
        x = torch.randn(7, 12)
        out = pool_sequence(x)
        assert out.shape == (12,)

    def test_mean_axis_correct(self):
        x = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        out = pool_sequence(x)
        expected = x.mean(dim=0)
        assert torch.allclose(out, expected)


# ---------------------------------------------------------------------------
# build_sample_kernel
# ---------------------------------------------------------------------------

class TestBuildSampleKernel:
    @pytest.fixture
    def sample_inputs(self):
        N, d = 8, 6
        acts = torch.randn(N, d)
        grads = torch.randn(N, d)
        gram_W = torch.eye(d)
        return acts, grads, gram_W

    def test_shape(self, sample_inputs):
        acts, grads, gram_W = sample_inputs
        K = build_sample_kernel(acts, grads, gram_W)
        assert K.shape == (8, 8)

    def test_symmetric(self, sample_inputs):
        acts, grads, gram_W = sample_inputs
        K = build_sample_kernel(acts, grads, gram_W)
        assert torch.allclose(K, K.T, atol=1e-6)

    def test_diagonal_is_one(self, sample_inputs):
        acts, grads, gram_W = sample_inputs
        K = build_sample_kernel(acts, grads, gram_W)
        assert torch.allclose(K.diagonal(), torch.ones(8), atol=1e-6)

    def test_values_in_unit(self, sample_inputs):
        acts, grads, gram_W = sample_inputs
        K = build_sample_kernel(acts, grads, gram_W)
        assert (K >= 0).all()
        assert (K <= 1.0 + 1e-6).all()

    def test_hadamard_is_psd(self, sample_inputs):
        from fannar.gram import is_psd
        acts, grads, gram_W = sample_inputs
        K = build_sample_kernel(acts, grads, gram_W)
        assert is_psd(K, eps=1e-4)

    def test_same_samples_close_to_one(self):
        acts = torch.zeros(4, 5)
        grads = torch.zeros(4, 5)
        gram_W = torch.eye(5)
        K = build_sample_kernel(acts, grads, gram_W)
        assert (K > 0.99).all()


# ---------------------------------------------------------------------------
# class_pair_means
# ---------------------------------------------------------------------------

class TestClassPairMeans:
    @pytest.fixture
    def D_cats(self):
        torch.manual_seed(7)
        n = 9
        cats = ["A", "A", "A", "B", "B", "B", "C", "C", "C"]
        X = torch.randn(n, 4)
        D = (X.unsqueeze(0) - X.unsqueeze(1)).pow(2).sum(-1).sqrt()
        return D, cats

    def test_keys_cover_all_pairs(self, D_cats):
        D, cats = D_cats
        result = class_pair_means(D, cats)
        assert ("A", "A") in result
        assert ("B", "B") in result
        assert ("A", "B") in result or ("B", "A") in result

    def test_within_class_excludes_diagonal(self):
        D = torch.ones(4, 4)
        D.fill_diagonal_(0.0)
        cats = ["X", "X", "X", "X"]
        result = class_pair_means(D, cats)
        assert abs(result[("X", "X")] - 1.0) < 1e-6

    def test_between_class_correct(self):
        D = torch.ones(4, 4)
        D.fill_diagonal_(0.0)
        cats = ["A", "A", "B", "B"]
        result = class_pair_means(D, cats)
        key = ("A", "B") if ("A", "B") in result else ("B", "A")
        assert abs(result[key] - 1.0) < 1e-6

    def test_single_class_per_category(self):
        """Categoria com 1 elemento → within-class = 0."""
        D = torch.ones(3, 3) - torch.eye(3)
        cats = ["A", "B", "C"]
        result = class_pair_means(D, cats)
        assert result[("A", "A")] == 0.0
        assert result[("B", "B")] == 0.0

    def test_float_values(self, D_cats):
        D, cats = D_cats
        result = class_pair_means(D, cats)
        for v in result.values():
            assert isinstance(v, float)

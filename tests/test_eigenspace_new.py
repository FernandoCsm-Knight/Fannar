"""Testes de kernel_pca e classical_mds (fannar.gram.eigenspace)."""

from __future__ import annotations

import pytest
import torch

from fannar.gram.eigenspace import classical_mds, kernel_pca
from fannar.gram.distance import distance_matrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gram_8():
    torch.manual_seed(42)
    X = torch.randn(8, 5)
    return X @ X.T


@pytest.fixture
def dist_8(gram_8):
    return distance_matrix(gram_8)


# ---------------------------------------------------------------------------
# kernel_pca
# ---------------------------------------------------------------------------

class TestKernelPCA:
    def test_coords_shape(self, gram_8):
        coords, lam, _ = kernel_pca(gram_8)
        assert coords.shape == (8, 2)

    def test_eigenvalues_shape(self, gram_8):
        _, lam, _ = kernel_pca(gram_8)
        assert lam.shape == (8,)

    def test_eigenvalues_nonneg(self, gram_8):
        _, lam, _ = kernel_pca(gram_8)
        assert (lam >= -1e-7).all()

    def test_eigenvalues_descending(self, gram_8):
        _, lam, _ = kernel_pca(gram_8)
        assert (lam[:-1] >= lam[1:] - 1e-6).all()

    def test_explained_in_unit(self, gram_8):
        _, _, explained = kernel_pca(gram_8)
        assert 0.0 <= explained <= 1.0 + 1e-8

    def test_explained_monotone_with_components(self, gram_8):
        _, _, exp2 = kernel_pca(gram_8, n_components=2)
        _, _, exp4 = kernel_pca(gram_8, n_components=4)
        assert exp4 >= exp2 - 1e-8

    def test_n_components_1(self, gram_8):
        coords, _, _ = kernel_pca(gram_8, n_components=1)
        assert coords.shape == (8, 1)

    def test_constant_kernel_collapses_to_zero(self):
        """Gram constante → HKH = 0 → coordenadas ≈ 0."""
        K = torch.ones(6, 6)
        coords, lam, explained = kernel_pca(K)
        assert lam.abs().max() < 1e-6

    def test_centering_removes_mean(self, gram_8):
        coords, _, _ = kernel_pca(gram_8)
        assert coords.mean(dim=0).abs().max() < 1e-5

    def test_identity_gram_spreads_points(self):
        K = torch.eye(8)
        coords, _, _ = kernel_pca(K)
        # pontos distintos têm coordenadas distintas
        assert coords.std() > 1e-6


# ---------------------------------------------------------------------------
# classical_mds
# ---------------------------------------------------------------------------

class TestClassicalMDS:
    def test_coords_shape(self, dist_8):
        coords, _ = classical_mds(dist_8)
        assert coords.shape == (8, 2)

    def test_stress_in_unit(self, dist_8):
        _, stress = classical_mds(dist_8)
        assert 0.0 <= stress <= 1.0 + 1e-8

    def test_zero_distance_matrix_zero_stress(self):
        D = torch.zeros(5, 5)
        _, stress = classical_mds(D)
        assert stress < 1e-6

    def test_n_components_1(self, dist_8):
        coords, _ = classical_mds(dist_8, n_components=1)
        assert coords.shape == (8, 1)

    def test_stress_decreases_with_components(self, dist_8):
        _, s1 = classical_mds(dist_8, n_components=1)
        _, s2 = classical_mds(dist_8, n_components=2)
        _, s3 = classical_mds(dist_8, n_components=3)
        assert s3 <= s2 + 1e-6
        assert s2 <= s1 + 1e-6

    def test_symmetric_input_symmetric_coords(self, dist_8):
        """MDS sobre D simétrica: coordenadas consistentes com simetria."""
        coords, _ = classical_mds(dist_8)
        D_rec = (coords.unsqueeze(0) - coords.unsqueeze(1)).pow(2).sum(-1).sqrt()
        assert torch.allclose(D_rec, D_rec.T, atol=1e-6)

    def test_all_points_same_gives_zero_coords(self):
        """Todos os pontos a distância zero → coordenadas ≈ zero."""
        D = torch.zeros(6, 6)
        coords, _ = classical_mds(D)
        assert coords.abs().max() < 1e-6

    def test_stress_two_points(self):
        """2 pontos: MDS 1D deve ter stress 0."""
        D = torch.tensor([[0.0, 3.0], [3.0, 0.0]])
        _, stress = classical_mds(D, n_components=1)
        assert stress < 1e-5

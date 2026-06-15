"""Testes de fannar.pipelines.token_geometry."""

from __future__ import annotations

import pytest
import torch

from fannar.pipelines.token_geometry import (
    cumulative_token_distances,
    token_distance_matrix,
)
from fannar.types import SpectralDecomposition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_decomp(d: int, seed: int = 0) -> SpectralDecomposition:
    """Decomposição espectral sintética de uma Gram d×d aleatória."""
    torch.manual_seed(seed)
    X = torch.randn(d, d)
    K = X @ X.T
    K = 0.5 * (K + K.T)
    lam, V = torch.linalg.eigh(K)
    lam = lam.flip(0)
    V = V.flip(1)
    return SpectralDecomposition(eigenvalues=lam, eigenvectors=V)


def _token_dist(S: int = 6, d: int = 8, seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    acts = torch.randn(S, d)
    dec = _make_decomp(d)
    return token_distance_matrix(acts, decomp_W=dec)


# ---------------------------------------------------------------------------
# token_distance_matrix
# ---------------------------------------------------------------------------

class TestTokenDistanceMatrix:
    def test_shape(self):
        D = _token_dist(S=6, d=8)
        assert D.shape == (6, 6)

    def test_diagonal_zero(self):
        D = _token_dist()
        assert (D.diagonal() < 1e-6).all()

    def test_symmetric(self):
        D = _token_dist()
        assert torch.allclose(D, D.T, atol=1e-5)

    def test_nonneg(self):
        D = _token_dist()
        assert (D >= 0).all()

    def test_with_decomp_A_and_G(self):
        S, d = 5, 6
        acts = torch.randn(S, d)
        dec_W = _make_decomp(d, seed=10)
        dec_A = _make_decomp(d, seed=11)
        dec_G = _make_decomp(d, seed=12)
        D = token_distance_matrix(acts, decomp_W=dec_W, decomp_A=dec_A, decomp_G=dec_G)
        assert D.shape == (S, S)
        assert (D.diagonal() < 1e-6).all()

    def test_only_W_vs_all_three_differ(self):
        S, d = 5, 6
        torch.manual_seed(0)
        acts = torch.randn(S, d)
        dec = _make_decomp(d)
        D_W = token_distance_matrix(acts, decomp_W=dec)
        D_all = token_distance_matrix(acts, decomp_W=dec, decomp_A=dec, decomp_G=dec)
        # M maior → distâncias maiores ou iguais
        assert (D_all >= D_W - 1e-5).all()

    def test_identical_tokens_zero_dist(self):
        d = 6
        acts = torch.ones(5, d)
        dec = _make_decomp(d)
        D = token_distance_matrix(acts, decomp_W=dec)
        assert (D < 1e-6).all()


# ---------------------------------------------------------------------------
# cumulative_token_distances
# ---------------------------------------------------------------------------

class TestCumulativeTokenDistances:
    def _layers(self, S: int = 5, d: int = 6, n_layers: int = 4) -> list[torch.Tensor]:
        result = []
        for seed in range(n_layers):
            result.append(_token_dist(S=S, d=d, seed=seed))
        return result

    def test_list_length(self):
        layers = self._layers(n_layers=4)
        out = cumulative_token_distances(layers)
        assert len(out) == 4

    def test_shapes(self):
        S = 5
        layers = self._layers(S=S, n_layers=3)
        out = cumulative_token_distances(layers)
        for D in out:
            assert D.shape == (S, S)

    def test_diagonal_zero(self):
        layers = self._layers()
        out = cumulative_token_distances(layers)
        for D in out:
            assert (D.diagonal() < 1e-6).all()

    def test_symmetric(self):
        layers = self._layers()
        out = cumulative_token_distances(layers)
        for D in out:
            assert torch.allclose(D, D.T, atol=1e-5)

    def test_nonneg(self):
        layers = self._layers()
        out = cumulative_token_distances(layers)
        for D in out:
            assert (D >= 0).all()

    def test_dict_input(self):
        layers = self._layers(n_layers=3)
        d_dict = {i: l for i, l in enumerate(layers)}
        list_out = cumulative_token_distances(layers)
        dict_out = cumulative_token_distances(d_dict)
        for a, b in zip(list_out, dict_out):
            assert torch.allclose(a, b, atol=1e-6)

    def test_single_layer_passthrough(self):
        """1 camada: a saída deve ser D derivado da afinidade do próprio D."""
        D0 = _token_dist(S=5, d=6, seed=42)
        out = cumulative_token_distances([D0])
        # D deve ser não-negativo, simétrico, diagonal zero
        D_cum = out[0]
        assert (D_cum.diagonal() < 1e-6).all()
        assert torch.allclose(D_cum, D_cum.T, atol=1e-5)

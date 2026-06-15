"""Smoke tests de visualização (backend Agg — sem janela gráfica)."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # força backend não-interativo antes de qualquer import do plt

import pytest
import torch

import matplotlib.pyplot as plt

from fannar.gram.distance import distance_matrix, sentence_distance_matrix
from fannar.pipelines.sample_geometry import build_sample_kernel
from fannar.pipelines.token_geometry import cumulative_token_distances, token_distance_matrix
from fannar.types import SpectralDecomposition
from fannar.visualization import (
    plot_rkhs_spectral_projection_2d,
    plot_sentence_distance_diagnostics,
    plot_token_trajectories,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gram(n: int, d: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    X = torch.randn(n, d)
    return X @ X.T


def _sample_kernels(N: int = 8, d: int = 6, n_layers: int = 6) -> dict[int, torch.Tensor]:
    kernels = {}
    for li in range(n_layers):
        torch.manual_seed(li)
        acts = torch.randn(N, d)
        grads = torch.randn(N, d)
        gram_W = torch.eye(d)
        kernels[li] = build_sample_kernel(acts, grads, gram_W)
    return kernels


def _decomp(d: int, seed: int = 0) -> SpectralDecomposition:
    torch.manual_seed(seed)
    X = torch.randn(d, d)
    K = X @ X.T
    K = 0.5 * (K + K.T)
    lam, V = torch.linalg.eigh(K)
    lam = lam.flip(0)
    V = V.flip(1)
    return SpectralDecomposition(eigenvalues=lam, eigenvectors=V)


# ---------------------------------------------------------------------------
# plot_rkhs_spectral_projection_2d
# ---------------------------------------------------------------------------

class TestPlotRKHSSpectralProjection2D:
    def test_returns_figure(self):
        N = 6
        labels = [f"s{i}" for i in range(N)]
        cats = ["A", "A", "B", "B", "C", "C"]
        kernels = _sample_kernels(N=N, n_layers=6)
        fig = plot_rkhs_spectral_projection_2d(kernels, labels, categories=cats)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_six_axes(self):
        N = 6
        labels = [f"s{i}" for i in range(N)]
        kernels = _sample_kernels(N=N, n_layers=6)
        fig = plot_rkhs_spectral_projection_2d(kernels, labels)
        assert len(fig.axes) >= 6
        plt.close(fig)

    def test_fewer_than_six_layers(self):
        N = 4
        labels = [f"s{i}" for i in range(N)]
        kernels = _sample_kernels(N=N, n_layers=3)
        fig = plot_rkhs_spectral_projection_2d(kernels, labels)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_custom_layer_indices(self):
        N = 8
        labels = [f"x{i}" for i in range(N)]
        kernels = _sample_kernels(N=N, n_layers=10)
        fig = plot_rkhs_spectral_projection_2d(
            kernels, labels, layer_indices=[0, 2, 4, 6, 8, 9]
        )
        assert hasattr(fig, "savefig")
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_sentence_distance_diagnostics
# ---------------------------------------------------------------------------

class TestPlotSentenceDistanceDiagnostics:
    def test_returns_figure(self):
        N = 6
        labels = [f"s{i}" for i in range(N)]
        cats = ["A", "A", "B", "B", "C", "C"]
        kernels = _sample_kernels(N=N, n_layers=4)
        fig = plot_sentence_distance_diagnostics(
            kernels, labels, cats, layer_idx=0
        )
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_six_subplots(self):
        N = 6
        labels = [f"s{i}" for i in range(N)]
        cats = ["A", "A", "B", "B", "C", "C"]
        kernels = _sample_kernels(N=N, n_layers=4)
        fig = plot_sentence_distance_diagnostics(
            kernels, labels, cats, layer_idx=0
        )
        # 2×3 grid = 6 axes (mais colorbars adicionam mais)
        assert len(fig.axes) >= 6
        plt.close(fig)

    def test_custom_n_levels(self):
        N = 4
        labels = [f"t{i}" for i in range(N)]
        cats = ["A", "A", "B", "B"]
        kernels = _sample_kernels(N=N, n_layers=3)
        fig = plot_sentence_distance_diagnostics(
            kernels, labels, cats, layer_idx=0, n_levels=6
        )
        assert hasattr(fig, "savefig")
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_token_trajectories
# ---------------------------------------------------------------------------

class TestPlotTokenTrajectories:
    def _cum_dists(self, S: int = 7, d: int = 6, n_layers: int = 6) -> list[torch.Tensor]:
        dists = []
        dec = _decomp(d)
        for seed in range(n_layers):
            torch.manual_seed(seed)
            acts = torch.randn(S, d)
            D = token_distance_matrix(acts, decomp_W=dec)
            dists.append(D)
        return cumulative_token_distances(dists)

    def test_returns_figure(self):
        S = 7
        cum = self._cum_dists(S=S)
        labels = [f"tok{i}" for i in range(S)]
        fig = plot_token_trajectories(cum, labels)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_six_axes(self):
        cum = self._cum_dists()
        labels = [f"w{i}" for i in range(7)]
        fig = plot_token_trajectories(cum, labels)
        assert len(fig.axes) >= 6
        plt.close(fig)

    def test_fewer_panels(self):
        cum = self._cum_dists(n_layers=3)
        labels = [f"t{i}" for i in range(7)]
        fig = plot_token_trajectories(cum, labels)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_custom_colors(self):
        cum = self._cum_dists(S=4, n_layers=6)
        labels = ["a", "b", "c", "d"]
        colors = ["red", "blue", "green", "orange"]
        fig = plot_token_trajectories(cum, labels, token_colors=colors)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_with_layer_indices(self):
        cum = self._cum_dists(n_layers=8)
        labels = [f"t{i}" for i in range(7)]
        fig = plot_token_trajectories(cum, labels, layer_indices=list(range(8)))
        assert hasattr(fig, "savefig")
        plt.close(fig)

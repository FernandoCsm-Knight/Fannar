"""Plots básicos do espectro geométrico."""

from __future__ import annotations

from typing import Any

import torch


def _np(t: torch.Tensor):
    return t.detach().cpu().numpy()


def plot_spectrum(eigenvalues: torch.Tensor, ax: Any = None, log: bool = True):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    lam = _np(eigenvalues.clamp_min(0))
    ax.plot(range(1, len(lam) + 1), lam, marker="o", ms=3)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("índice do modo")
    ax.set_ylabel("autovalor")
    ax.set_title("Espectro da Gram")
    return ax

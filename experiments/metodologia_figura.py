"""Figura de metodologia: as matrizes de Gram, etapa por etapa (`paper/figuras/gram_construcao.pdf`).

Mostra a construcao da Subsecao `subsec:construcao`: as tres Grams componentes
G_l = (T_a(G~_l) + J)/2, o produto de Hadamard que as compoe e a Gram conjunta depois de
T_tr ∘ T_c. A base e a `segment` (7 classes), com o MLP residual da suite e alvo fixo; poucas
classes deixam os blocos de classe grandes o bastante para serem vistos. A figura e ilustrativa
(20 amostras por classe); os resultados numericos do artigo usam sempre a amostra de ~500.

Uso:
    python metodologia_figura.py --dataset segment --block b3
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import seaborn as sns
import torch

from src import theme as T
from src.data_tabular import balanced_sample, load_tabular, split, standardize
from src.mlp import ResidualMLP
from tabular_suite import extract, train

T.use()

import matplotlib.pyplot as plt  # noqa: E402,F401

STAGES = [("params", r"$\mathbf{G}_W$"), ("activations", r"$\mathbf{G}_A$"), ("grads", r"$\mathbf{G}_\Gamma$")]


def component(u: np.ndarray) -> np.ndarray:
    """(T_a(G~) + J)/2 com o kernel linear: o cosseno deslocado para [0, 1]."""
    u = u.reshape(len(u), -1).astype(np.float64)
    n = np.maximum(np.linalg.norm(u, axis=1), 1e-300)
    return ((u @ u.T) / np.outer(n, n) + 1.0) / 2.0


def centered_trace(G: np.ndarray) -> np.ndarray:
    H = np.eye(len(G)) - 1.0 / len(G)
    C = H @ G @ H
    return C / np.trace(C)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="segment")
    ap.add_argument("--block", default="b3")
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-target", type=int, default=0)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--out", default="paper/figuras/gram_construcao.pdf")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = load_tabular(args.dataset)
    tr, va, te = split(y, args.seed)
    X_tr, X_va, X_te = (a.astype(np.float32) for a in standardize(X[tr], X[va], X[te]))
    torch.manual_seed(args.seed)
    model = ResidualMLP(X.shape[1], int(y.max() + 1), args.width, args.blocks).to(device)
    train(model, X_tr, y[tr], X_va, y[va], device, args, args.seed)
    model.eval()

    idx = balanced_sample(y[te], total=args.per_class * int(y.max() + 1), seed=args.seed)
    order = idx[np.argsort(y[te][idx], kind="stable")]
    labels = y[te][order]
    src = extract(model, X_te[order], device, args.fixed_target)["sources"][args.block]
    comps = [component(src[key]) for key, _ in STAGES]
    product = comps[0] * comps[1] * comps[2]
    panels = list(zip(comps, [t for _, t in STAGES])) + [
        (product, r"$\mathbf{G}_W \circ \mathbf{G}_A \circ \mathbf{G}_\Gamma$"),
        (centered_trace(product), r"$T_{\mathrm{tr}}(T_c(\cdot))$")]
    bounds = np.flatnonzero(np.diff(labels)) + 1

    fig, axes = T.grid(1, len(panels), width="full", height=1.75)
    for ax, (m, title) in zip(axes[0], panels):
        lo, hi = np.percentile(m, [1, 99])
        sns.heatmap(m, cmap=T.SEQ, vmin=lo, vmax=hi, ax=ax, square=True, cbar=False,
                    xticklabels=False, yticklabels=False)
        for b in bounds:
            ax.axhline(b, color=T.SURFACE, lw=0.4)
            ax.axvline(b, color=T.SURFACE, lw=0.4)
        ax.set_title(title, fontsize=8.0)
        ax.set_xlabel(f"[{lo:.2g}, {hi:.2g}]".replace(".", ","), fontsize=5.8, labelpad=1.5, color=T.INK_FAINT)
    T.panel_tags(axes)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    T.save(fig, args.out)
    off = ~np.eye(len(labels), dtype=bool)
    med = {t: float(np.median(c[off])) for c, (_, t) in zip(comps, STAGES)}
    print(f"{len(labels)} amostras, bloco {args.block}; medianas fora da diagonal {med}, "
          f"produto {np.median(product[off]):.3f} -> {args.out}")


if __name__ == "__main__":
    main()

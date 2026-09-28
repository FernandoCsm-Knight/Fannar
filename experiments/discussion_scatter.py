"""Figura da discussao: a estrutura geometrica do espaco de representacao, bloco a bloco.

Como a Gram G e semidefinida positiva, os objetos admitem uma realizacao φ_i com
G_ij = ⟨φ_i, φ_j⟩ (eq. `empirical_feature_realization`). As duas coordenadas mostradas sao as
dos dois modos dominantes, φ_i ≈ (√λ1 V_i1, √λ2 V_i2): uma PCA de nucleo da propria Gram, sem
parametros nem as distorcoes de metodos nao lineares. Cada painel informa a fracao da energia
(traco) retida pelos dois eixos e o acerto do k-vizinhos contra a predicao, calculado na Gram
inteira, e nao no plano.

Base `segment` (7 classes), todo o conjunto de teste, MLP residual da suite (semente 0), alvo
fixo. Linhas: Gram conjunta da rede treinada, so as ativacoes da rede treinada, Gram conjunta da
rede nao treinada (o piso). Cor = classe predita pela rede treinada; x = erro da rede treinada.

Uso:
    python discussion_scatter.py
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from src import theme as T
from src.behaviour import knn_loo
from src.data_tabular import class_names, load_tabular, split, standardize
from src.mlp import ResidualMLP
from tabular_suite import extract, train

T.use()

import matplotlib.pyplot as plt  # noqa: E402,F401


def component(u: np.ndarray) -> np.ndarray:
    u = u.reshape(len(u), -1).astype(np.float64)
    g = u @ u.T
    s = np.sqrt(np.maximum(np.diag(g), 1e-300))
    return (g / np.outer(s, s) + 1.0) / 2.0


def centered_trace(g: np.ndarray) -> np.ndarray:
    r = g.mean(1, keepdims=True)
    c = g - r - r.T + g.mean()
    return c / np.trace(c)


def joint(src: dict) -> np.ndarray:
    return centered_trace(component(src["params"]) * component(src["activations"]) * component(src["grads"]))


def linear(u: np.ndarray) -> np.ndarray:
    u = u.reshape(len(u), -1).astype(np.float64)
    return centered_trace(u @ u.T)


def embed(g: np.ndarray):
    vals, vecs = np.linalg.eigh(g)
    idx = np.argsort(vals)[::-1][:2]
    lam = np.maximum(vals[idx], 0)
    return vecs[:, idx] * np.sqrt(lam), float(lam.sum() / np.trace(g))


def distances(g: np.ndarray) -> np.ndarray:
    d = np.diag(g)
    return np.sqrt(np.maximum(d[:, None] + d[None, :] - 2 * g, 0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="segment")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-target", type=int, default=0)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--out", default="outputs/discussion/scatter.pdf")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = load_tabular(args.dataset)
    tr, va, te = split(y, args.seed)
    X_tr, X_va, X_te = (a.astype(np.float32) for a in standardize(X[tr], X[va], X[te]))
    torch.manual_seed(args.seed)
    model = ResidualMLP(X.shape[1], int(y.max() + 1), args.width, args.blocks).to(device)
    untrained = copy.deepcopy(model)
    train(model, X_tr, y[tr], X_va, y[va], device, args, args.seed)
    model.eval()
    untrained.eval()
    ext = extract(model, X_te, device, args.fixed_target)
    ext_u = extract(untrained, X_te, device, args.fixed_target)
    preds, y_te = ext["preds"], y[te]
    wrong = preds != y_te
    classes = class_names(args.dataset)
    blocks = model.block_names
    print(f"{len(y_te)} amostras de teste, acerto {1 - wrong.mean():.3f}")

    rows = [("conjunta, rede treinada", lambda b: joint(ext["sources"][b])),
            # a fonte isolada passa pela mesma transformacao das componentes (angular, deslocada,
            # centrada e normalizada em traco): a linha compara composicao × uma fonte, e nao kernels
            ("só ativações, rede treinada", lambda b: centered_trace(component(ext["sources"][b]["activations"]))),
            ("conjunta, rede não treinada", lambda b: joint(ext_u["sources"][b]))]
    import seaborn as sns
    # colorblind (Okabe-Ito) com 10 cores, sem o rosa (baixo contraste) e o cinza (reservado ao controle)
    colors = [sns.color_palette("colorblind", 10)[i] for i in (0, 1, 2, 3, 4, 5, 9)]
    fig, axes = T.grid(len(rows), len(blocks), width="full", height=3.55)
    stats = {}
    for r, (name, fn) in enumerate(rows):
        for c, b in enumerate(blocks):
            g = fn(b)
            xy, frac = embed(g)
            knn = knn_loo(distances(g), preds, 5)
            stats[(name, b)] = (frac, knn)
            ax = axes[r, c]
            for k in range(len(classes)):
                m = (preds == k) & ~wrong
                ax.scatter(xy[m, 0], xy[m, 1], s=1.6, color=colors[k], linewidths=0, alpha=0.8,
                           label=classes[k] if (r, c) == (0, 0) else None, rasterized=True)
            ax.scatter(xy[wrong, 0], xy[wrong, 1], s=6, marker="x", color=T.INK, linewidths=0.5,
                       label="erro do modelo" if (r, c) == (0, 0) else None, zorder=3)
            T.bare(ax)
            ax.text(0.03, 0.03, f"{frac:.0%} · kNN {knn:.2f}".replace(".", ","), transform=ax.transAxes,
                    fontsize=5.0, color=T.INK_SOFT, va="bottom",
                    bbox=dict(facecolor=T.SURFACE, edgecolor="none", pad=0.8, alpha=0.85))
            if r == 0:
                ax.set_title(b, fontsize=8)
            if c == 0:
                ax.set_ylabel(name.replace(", ", ",\n"), fontsize=6.0)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=len(classes) + 1,
               fontsize=6.0, frameon=False, markerscale=3.0, bbox_to_anchor=(0.5, -0.01), handletextpad=0.2, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    T.save(fig, args.out)
    for (name, b), (frac, knn) in stats.items():
        print(f"{name:30s} {b:5s} energia 2D {frac:.3f}  kNN {knn:.3f}")


if __name__ == "__main__":
    main()

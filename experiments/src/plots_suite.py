"""Figura das leituras do espaco de representacao por bloco (ver `src/theme.py`).

Cada painel e uma leitura ao longo dos blocos: uma linha fina por base (media entre
sementes), a media entre bases em destaque com a faixa de ± um desvio padrao entre bases, A
isolado e o RSA como comparacao, e o piso cruzado tracejado. O que se le e a *forma* da curva
repetida entre bases muito diferentes, e nao o valor de uma base em particular.
"""

from __future__ import annotations

import numpy as np

from . import theme as T

T.use()

import matplotlib.pyplot as plt  # noqa: E402,F401  (tema antes do pyplot)

PANELS = [
    ("knn_pred", "k-vizinhos contra a predição"),
    ("correctness_auc", "AUC da margem para o acerto"),
    ("confusion_soft", "Spearman com a confusão suave"),
    ("containment_k", r"contenção das classes em $V_r$ ($r = K-1$)"),
    ("e_ratio_auc_k", r"AUC de $\hat E_{\mathrm{ratio}}$ para o erro ($r = K-1$)"),
    ("cka_alignment", r"alinhamento $a$ com a CKA"),
    ("cka_halves", "AUC do acerto nas metades da CKA"),
]
WITH_RSA = {"knn_pred", "correctness_auc", "confusion_soft"}


def _curve(summary, key):
    """Media e desvio da leitura. Com varias bases, o desvio e entre bases; com uma so (o Pet),
    e entre sementes -- de outro modo a faixa teria largura zero e esconderia a variacao."""
    if len(summary["datasets"]) == 1:
        m, s = summary["per_dataset"][summary["datasets"][0]][key]
        return (np.asarray(m, dtype=float), np.asarray(s, dtype=float))
    return tuple(np.asarray(v, dtype=float) for v in summary["across"][key][:2])


def plot_suite(summary: dict, path: str) -> None:
    blocks, names = summary["blocks"], summary["datasets"]
    x = np.arange(len(blocks))
    fig, axes = T.grid(2, 4, width="page", height=4.6)
    color = T.COLOR["joint"]
    for ax, (key, title) in zip(axes.ravel(), PANELS):
        if key == "cka_halves":
            for part, style, label in (("cka_resid_auc", "joint", "metade residual"),
                                       ("cka_seen_auc", "rsa_euclid", "metade vista pela CKA")):
                m, s = _curve(summary, f"{part}/joint")
                ax.fill_between(x, m - s, m + s, color=T.COLOR[style], alpha=0.14, linewidth=0)
                ax.plot(x, m, lw=1.5, ms=3.2, **{**T.series(style), "label": label})
            ax.legend(fontsize=6.3, loc="lower right")
        else:
            for n in names:
                m = np.asarray(summary["per_dataset"][n][f"{key}/joint"][0], dtype=float)
                ax.plot(x, m, color=color, lw=0.5, alpha=0.3)
            m, s = _curve(summary, f"{key}/joint")
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.18, linewidth=0)
            ax.plot(x, m, lw=1.6, ms=3.4, **T.series("joint"))
            if key != "cka_alignment":
                ax.plot(x, _curve(summary, f"{key}/floor")[0], lw=1.0, ms=3.0,
                        **T.control("piso cruzado"))
                ax.plot(x, _curve(summary, f"{key}/activations")[0], lw=1.0, ms=3.0,
                        **T.series("activations"))
            if key in WITH_RSA:
                ax.plot(x, _curve(summary, f"{key}/rsa")[0], lw=1.0, ms=3.0,
                        **T.series("rsa_euclid"))
        ax.set_title(title, fontsize=8.0)
        ax.set_xticks(x)
        ax.set_xticklabels(blocks)
        ax.set_xlim(-0.35, len(blocks) - 0.65)
        T.despine(ax)
        T.decimal(ax, "y", 1)
    axes.ravel()[-1].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=4, fontsize=7.0,
               bbox_to_anchor=(0.5, -0.03))
    T.panel_tags([a for a in axes.ravel() if a.get_visible()])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    T.save(fig, path)

"""Standard figures. Every function returns the ``matplotlib.figure.Figure`` it draws.

Conventions shared by all figures: a colour-blind-safe palette (Okabe-Ito), one fixed marker
per series, the floor always grey and dashed, and titles/labels that say what is measured.
"""

from __future__ import annotations

import numpy as np

from .readings import auc  # noqa: F401  (re-exported for convenience in notebooks)
from .report import LABELS, Report

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
FLOOR = dict(color="0.55", linestyle=(0, (4, 2.5)), marker="s", markerfacecolor="white")


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)


def _class_colors(n: int) -> list:
    if n <= len(OKABE_ITO):
        return OKABE_ITO[:n]
    cmap = _plt().get_cmap("tab20" if n <= 20 else "turbo")
    return [cmap(t) for t in np.linspace(0, 1, n)]


# ---------------------------------------------------------------------- readings
def plot_report(report: Report, readings=None):
    """One panel per reading, one line per representation, layers on the x axis. Readings that are
    undefined everywhere (e.g. the confusion agreement with fewer than 5 classes) are left out."""
    plt = _plt()
    readings = [r for r in (readings or report.readings) if r != "rank"
                and any(np.isfinite(report.series(r, rep)).any() for rep in report.representations)]
    ncol = min(3, len(readings))
    nrow = int(np.ceil(len(readings) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow), squeeze=False)
    x = np.arange(len(report.layers))
    reps = [r for r in report.representations if r != "floor"]
    for ax, reading in zip(axes.ravel(), readings):
        for t, rep in enumerate(reps):
            ax.plot(x, report.series(reading, rep), color=OKABE_ITO[t % 8], marker=MARKERS[t % 8], ms=4,
                    lw=1.8 if rep == "joint" else 1.2, label=rep)
        if "floor" in report.values:
            ax.plot(x, report.series(reading, "floor"), lw=1.1, ms=4, label="floor", **FLOOR)
        ax.set_xticks(x)
        ax.set_xticklabels(report.layers, rotation=45 if len(x) > 8 else 0, fontsize=7)
        ax.set_title(LABELS.get(reading, reading), fontsize=9)
        _style(ax)
    for ax in axes.ravel()[len(readings):]:
        ax.set_visible(False)
    axes.ravel()[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------- geometry
def plot_embedding(rows: dict, layers: list, colors, labels=None, predictions=None, class_names=None):
    """The space of each layer projected on its two dominant modes (kernel PCA of G).

    ``rows`` maps a row name to one :class:`GramSpace` per layer. Colour = ``colors`` (predicted
    class by default); ``x`` marks the model's errors when labels are known."""
    plt = _plt()
    colors = np.asarray(colors)
    classes = np.unique(colors)
    palette = dict(zip(classes, _class_colors(len(classes))))
    wrong = None if labels is None or predictions is None else np.asarray(predictions) != np.asarray(labels)
    fig, axes = plt.subplots(len(rows), len(layers), figsize=(1.9 * len(layers), 1.9 * len(rows) + 0.5), squeeze=False)
    for r, (name, spaces) in enumerate(rows.items()):
        for c, sp in enumerate(spaces):
            ax = axes[r, c]
            xy, frac = sp.embedding(2)
            ok = np.ones(len(xy), bool) if wrong is None else ~wrong
            for cls in classes:
                m = (colors == cls) & ok
                ax.scatter(xy[m, 0], xy[m, 1], s=4, color=palette[cls], linewidths=0, alpha=0.85)
            if wrong is not None:
                ax.scatter(xy[wrong, 0], xy[wrong, 1], s=10, marker="x", color="black", linewidths=0.6)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.02, 0.02, f"{frac:.0%}", transform=ax.transAxes, fontsize=6, color="0.35")
            if r == 0:
                ax.set_title(str(layers[c]), fontsize=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=8)
    handles = [plt.Line2D([], [], ls="none", marker="o", ms=4, color=palette[c],
                          label=(class_names[c] if class_names is not None and np.issubdtype(type(c), np.integer) else str(c)))
               for c in classes[:20]]
    if wrong is not None:
        handles.append(plt.Line2D([], [], ls="none", marker="x", color="black", label="model error"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 11), fontsize=7, frameon=False,
               handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def plot_gram(space, order_by=None):
    """Component Grams, their Hadamard product and the final Gram, objects sorted by ``order_by``."""
    plt = _plt()
    mats = list(space.components.items()) + [("G", space.G)]
    order = np.argsort(np.asarray(order_by), kind="stable") if order_by is not None else np.arange(space.n)
    bounds = np.flatnonzero(np.diff(np.asarray(order_by)[order])) + 1 if order_by is not None else []
    if len(space.components) > 1:
        prod = np.prod(np.stack(list(space.components.values())), axis=0)
        mats.insert(len(space.components), ("Hadamard", prod))
    fig, axes = plt.subplots(1, len(mats), figsize=(2.4 * len(mats), 2.6), squeeze=False)
    for ax, (name, m) in zip(axes[0], mats):
        m = m[np.ix_(order, order)]
        lo, hi = np.percentile(m, [1, 99])
        ax.imshow(m, cmap="Blues", vmin=lo, vmax=hi, interpolation="nearest")
        for b in bounds:
            ax.axhline(b - 0.5, color="white", lw=0.4); ax.axvline(b - 0.5, color="white", lw=0.4)
        ax.set_title(name, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_spectrum(space, max_modes: int = 50):
    """Eigenvalues (fraction of the trace), cumulative energy, and the participation rank."""
    plt = _plt()
    vals = space.eigenvalues
    frac = vals / max(vals.sum(), 1e-300)
    k = min(max_modes, len(vals))
    r = space.participation_rank()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    axes[0].bar(np.arange(1, k + 1), frac[:k], color=OKABE_ITO[0])
    axes[0].set_xlabel("mode", fontsize=8); axes[0].set_ylabel("fraction of the trace", fontsize=8)
    axes[1].plot(np.arange(1, k + 1), np.cumsum(frac)[:k], color=OKABE_ITO[0], marker="o", ms=3)
    for ax in axes:
        ax.axvline(r + 0.5, color="0.4", ls=":", lw=1)
        _style(ax)
    axes[1].text(r + 1, 0.05, f"participation rank r = {r}", fontsize=7, color="0.3")
    axes[1].set_xlabel("modes", fontsize=8); axes[1].set_ylabel("cumulative energy", fontsize=8)
    axes[1].set_ylim(0, 1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------- pairs and maps
def plot_pair(expl, k: int = 8):
    """Top-``k`` features of a pair: share per layer (heatmap) and the values in ``i`` and ``j``."""
    plt = _plt()
    order = expl.ranking()[:k]
    names = expl.feature_names or [f"x{t}" for t in range(len(expl.xi))]
    shares = np.stack([expl.shares(layer)[order] for layer in expl.layers])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.35 * k + 1.3), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    im = ax.imshow(shares.T, cmap="Blues", vmin=0, vmax=max(shares.max(), 1e-12), aspect="auto")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names[f] for f in order], fontsize=8)
    ax.set_xticks(range(len(expl.layers))); ax.set_xticklabels(expl.layers, fontsize=7, rotation=45)
    ax.set_title("share of the gain in proximity, per layer", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax = axes[1]
    y = np.arange(len(order))
    ax.barh(y - 0.18, expl.xi[order], height=0.35, color=OKABE_ITO[0], label=f"i (pred {expl.pred_i})")
    ax.barh(y + 0.18, expl.xj[order], height=0.35, color=OKABE_ITO[1], label=f"j (pred {expl.pred_j})")
    ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis()
    ax.set_title("feature values", fontsize=9); ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_errors(expl, k: int | None = None, layer: str | None = None, rows: str = "class"):
    """Which features the model's errors hinge on, by true class (``rows="class"``) or by
    true→predicted confusion (``rows="confusion"``).

    * matrix: columns are the features (the ``k`` with the largest mean share over all errors),
      cells the mean share of the feature among the row's errors, with an arrow for the direction
      (↑: the errors have lower values than the nearest correct object of their true class; ↓:
      higher);
    * top: mean share over the errors against the mean share over correctly predicted objects
      (how much the feature matters to the model in general);
    * right: errors per row, and the error rate of the class (or the fraction of all errors).
    """
    plt = _plt()
    layer = layer or expl.layers[-1]
    cols = expl.ranking(layer)[:k]
    names = [expl.names[f] for f in cols]
    cname = (lambda c: str(expl.class_names[c])) if expl.class_names else (lambda c: f"class {c}")
    if rows == "class":
        keys = sorted(expl.support) or sorted(set(expl.true.tolist()))
        masks = [expl.true == c for c in keys]
        labels = [cname(c) for c in keys]
        totals = [expl.support.get(c, int(m.sum())) for c, m in zip(keys, masks)]
    elif rows == "confusion":
        confs = expl.confusions()
        masks = [(expl.true == a) & (expl.pred == b) for a, b, _ in confs]
        labels = [f"{cname(a)} → {cname(b)}" for a, b, _ in confs]
        totals = [len(expl.errors)] * len(confs)
    else:
        raise ValueError("rows must be 'class' or 'confusion'")
    mat = np.full((len(masks), len(cols)), np.nan)
    arrows = np.zeros_like(mat)
    for r, m in enumerate(masks):
        if m.any():
            mat[r], arrows[r] = expl.mean_shares(layer, m)[cols], expl.direction(layer, m)[cols]

    nr, nc = len(masks), len(cols)
    fig = plt.figure(figsize=(max(7.5, 0.8 * nc + 4.2), 2.6 + 0.48 * nr))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.7, max(0.48 * nr, 1.2)], width_ratios=[0.8 * nc, 2.6],
                          hspace=0.08, wspace=0.05)
    ax_top, ax_note = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_mat = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_rate = fig.add_subplot(gs[1, 1], sharey=ax_mat)

    x = np.arange(nc)
    err_share, ref = expl.mean_shares(layer)[cols], expl.reference_shares(layer)
    width = 0.38 if ref is not None else 0.6
    ax_top.bar(x - (width / 2 if ref is not None else 0), err_share, width, color=OKABE_ITO[0], label="over the errors")
    if ref is not None:
        ax_top.bar(x + width / 2, ref[cols], width, color="0.75", label="over correct predictions")
    for t, v in enumerate(err_share):
        ax_top.text(t - (width / 2 if ref is not None else 0), v, f"{v:.0%}", ha="center", va="bottom", fontsize=7)
    top = max(err_share.max(), 0 if ref is None else ref[cols].max())
    ax_top.set_ylim(0, top * 1.25 + 1e-12)
    ax_top.set_ylabel("mean share", fontsize=8)
    ax_top.set_title(f"features behind the errors · layer {layer} · {len(expl.errors)} errors", fontsize=9, loc="left")
    ax_top.legend(fontsize=7, frameon=False, loc="upper right")
    ax_top.tick_params(labelbottom=False)
    _style(ax_top)

    cmap = _plt().get_cmap("Blues").with_extremes(bad="0.94")
    vmax = max(np.nanmax(mat) if np.isfinite(mat).any() else 0, 1e-12)
    ax_mat.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for r in range(nr):
        for c in range(nc):
            v = mat[r, c]
            if np.isnan(v):
                continue
            a = arrows[r, c]
            mark = (" ↑" if a > 0 else " ↓") if abs(a) >= 1 / 3 and v >= 0.02 else ""
            ax_mat.text(c, r, f"{v:.0%}{mark}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.6 * vmax else "0.15")
    ax_mat.set_xticks(x)
    ax_mat.set_xticklabels(names, fontsize=8, rotation=40 if max(map(len, names)) > 4 else 0,
                           ha="right" if max(map(len, names)) > 4 else "center")
    ax_mat.set_yticks(range(nr)); ax_mat.set_yticklabels(labels, fontsize=8)
    ax_mat.set_xticks(np.arange(nc + 1) - 0.5, minor=True); ax_mat.set_yticks(np.arange(nr + 1) - 0.5, minor=True)
    ax_mat.grid(which="minor", color="white", linewidth=1.5); ax_mat.tick_params(which="minor", length=0)
    for side in ax_mat.spines.values():
        side.set_visible(False)
    ax_mat.set_ylabel("true class" if rows == "class" else "true → predicted", fontsize=8)

    counts = np.array([int(m.sum()) for m in masks])
    rate = counts / np.maximum(totals, 1)
    ax_rate.barh(np.arange(nr), rate, height=0.6, color=OKABE_ITO[3])
    for r in range(nr):
        txt = f"{counts[r]}/{totals[r]} ({rate[r]:.0%})" if rows == "class" else f"{counts[r]} ({rate[r]:.0%})"
        ax_rate.text(rate[r], r, " " + txt, va="center", fontsize=7)
    ax_rate.set_xlim(0, min(max(rate.max(), 1e-12) * 1.9, 1.0))  # labels may run past 1: a rate never does
    ax_rate.set_xlabel("error rate" if rows == "class" else "share of the errors", fontsize=8)
    ax_rate.tick_params(labelleft=False, left=False)
    _style(ax_rate)

    ax_note.axis("off")
    ax_note.text(0.06, 0.95, "how to read\n"
                 "cell: share of the feature in what\n  separates the row's errors from\n"
                 "  the nearest correct object of\n  their true class\n"
                 "↑  errors have lower values\n↓  errors have higher values\n"
                 "grey bar: same share over correct\n  predictions (general relevance)",
                 va="top", fontsize=7, color="0.25", linespacing=1.35, transform=ax_note.transAxes)
    return fig


def plot_energy_map(energy: np.ndarray, image=None):
    """Retained-energy map of one input, optionally over the image (``(H, W[, 3])`` in [0, 1])."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    if image is not None:
        img = np.asarray(image)
        ax.imshow(img, extent=(0, 1, 0, 1))
        ax.imshow(energy, cmap="Blues", alpha=0.6, extent=(0, 1, 0, 1), interpolation="bilinear")
    else:
        ax.imshow(energy, cmap="Blues", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("retained spectral energy", fontsize=9)
    fig.tight_layout()
    return fig

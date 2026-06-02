"""Gráfico de radar do perfil explicativo de oito eixos."""

from __future__ import annotations

import math
from typing import Any

from ..types import ExplanatoryProfile


def plot_explanatory_radar(profile: ExplanatoryProfile, ax: Any = None, fill: float = 0.0):
    """Radar dos 8 eixos. Eixos indisponíveis usam ``fill`` (default 0)."""
    import matplotlib.pyplot as plt

    d = profile.as_dict()
    # Eixos indisponíveis marcados no próprio rótulo, sem texto flutuante sobreposto
    labels = [f"{name}*" if d[name] is None else name for name in ExplanatoryProfile.AXES]
    values = profile.as_vector(fill=fill).tolist()
    n = len(labels)
    angles = [i / n * 2 * math.pi for i in range(n)]
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]

    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)
    ax.plot(angles_closed, values_closed, linewidth=2)
    ax.fill(angles_closed, values_closed, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Perfil explicativo  (* = indisponível)")
    return ax

"""Visualizações interativas opcionais com plotly.

plotly é importado preguiçosamente; se não estiver instalado, levanta um erro
claro pedindo ``pip install fannar[viz]``.
"""

from __future__ import annotations

import torch

from ..gram.eigenspace import spectral_coordinates


def _require_plotly():
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plotly não instalado. Instale com `pip install fannar[viz]`."
        ) from exc
    import plotly.graph_objects as go
    return go


def interactive_projection_3d(K: torch.Tensor, labels: list[str] | None = None,
                              color: torch.Tensor | None = None):
    """Scatter 3D interativo das coordenadas espectrais."""
    go = _require_plotly()
    coords = spectral_coordinates(K, dim=3).detach().cpu().numpy()
    c = color.detach().cpu().numpy() if color is not None else None
    fig = go.Figure(data=[go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode="markers+text" if labels else "markers",
        text=labels,
        marker=dict(size=4, color=c, colorscale="Viridis", showscale=c is not None),
    )])
    fig.update_layout(title="Projeção espectral 3D (interativa)")
    return fig

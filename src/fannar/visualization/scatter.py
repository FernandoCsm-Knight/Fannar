"""Scatter 3D/2D interativo (Plotly) do espaço de representação induzido por kernel.

Uso central: avaliação de sobreposição semântica.

    z_i = (√λ₁ v_{i,1}, √λ₂ v_{i,2}, √λ₃ v_{i,3})

As coordenadas z_i satisfazem ⟨z_i, z_j⟩_eucl ≈ K(i,j) na projeção em dim
dimensões (MDS clássico sobre a Gram). Proximidade no scatter = similaridade
no espaço de representação do kernel. Se objetos de categorias distintas ficam
próximos, o modelo os representa como semanticamente similares.

Exemplo de uso::

    # N imagens divididas em categorias "animal" e "comida"
    A = activations.mean(dim=(-2, -1))       # (N, C) via pooling espacial
    K = CosineKernel()(A)                     # (N, N) Gram entre amostras
    fig = plot_semantic_scatter_3d(
        K, labels=nomes, categories=categorias
    )
    fig.show()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import plotly.graph_objects as go


# Paleta semântica por padrão (Seaborn-inspired, acessível)
_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#E07B54", "#76C893", "#B5179E", "#F48C06", "#4895EF",
]


def _palette(n: int) -> list[str]:
    return [_PALETTE[i % len(_PALETTE)] for i in range(n)]


def _center(K: torch.Tensor) -> torch.Tensor:
    """Duplo centramento H K H — remove componente médio, preserva estrutura relativa."""
    row = K.mean(dim=1, keepdim=True)
    col = K.mean(dim=0, keepdim=True)
    return K - row - col + K.mean()


def plot_semantic_scatter_3d(
    K: torch.Tensor,
    labels: list[str] | None = None,
    categories: list[str] | None = None,
    *,
    title: str = "Espaço de representação",
    center: bool = True,
    dim: int = 3,
    show_fidelity: bool = True,
    show_ellipsoids: bool = True,
    marker_size: int = 6,
    opacity: float = 0.85,
    width: int = 900,
    height: int = 720,
) -> "go.Figure":
    """Scatter interativo do espaço de representação induzido pelo kernel K.

    Cada ponto corresponde a um objeto (imagem, patch, canal, conceito...).
    As coordenadas são extraídas da decomposição espectral de K — elas
    representam o produto interno do RKHS como produto interno euclidiano
    em ``dim`` dimensões.

    Sobreposição semântica
    ----------------------
    Se pontos de categorias distintas (e.g. "animal" e "comida") ficam próximos
    no scatter, o modelo os representa de forma similar — indicando que o espaço
    de representação confunde ou associa essas categorias.

    Parâmetros
    ----------
    K : (N, N)
        Matriz de Gram entre os N objetos. Pode ser K_A (ativações),
        K_total (conjunta), ou qualquer kernel calculado sobre as
        representações de camada.
    labels : list[str] | None
        Nome de cada objeto (aparece no hover). Padrão: "obj_0", "obj_1", ...
    categories : list[str] | None
        Categoria semântica de cada objeto (define a cor). Padrão: "objects".
    center : bool
        Aplica H K H antes da projeção (padrão True). Recomendado: remove
        o componente médio e expõe a estrutura relativa entre categorias.
    dim : {2, 3}
        Dimensão da projeção. 3 (padrão) = scatter 3D interativo;
        2 = scatter 2D (mais fácil de comparar entre camadas).
    show_fidelity : bool
        Exibe no subtítulo a fidelidade de embedding: fração do traço de K
        capturada nas ``dim`` dimensões.
    show_ellipsoids : bool
        Adiciona elipsoides de covariância por categoria em 3D (apenas dim=3).
        Ajuda a quantificar o tamanho e a forma do cluster de cada categoria.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError(
            "plotly não encontrado. Instale com: pip install fannar[viz]"
        ) from e

    from ..gram.eigenspace import embedding_fidelity, spectral_coordinates

    n = K.shape[0]
    if labels is None:
        labels = [f"obj_{i}" for i in range(n)]
    if categories is None:
        categories = ["objects"] * n
    if dim not in (2, 3):
        raise ValueError(f"dim deve ser 2 ou 3, recebido {dim}.")

    K_work = K.detach().float()
    K_work = 0.5 * (K_work + K_work.T)
    if center:
        K_work = _center(K_work)

    coords = spectral_coordinates(K_work, dim=dim).cpu().numpy()   # (N, dim)
    fid = embedding_fidelity(K_work, dim=dim)

    # Kernel PCA com kernel cosseno: z_i satisfaz <z_i, z_j> ≈ K_c(i,j)
    method_tag = "Kernel PCA · kernel cosseno"
    sub = f"{method_tag} · variância explicada em {dim}D = {fid:.1%}" if show_fidelity else method_tag
    title_full = f"{title}<br><sup>{sub}</sup>"

    unique_cats = list(dict.fromkeys(categories))
    colors = _palette(len(unique_cats))
    cat_color = {c: colors[i] for i, c in enumerate(unique_cats)}

    fig = go.Figure()

    for cat in unique_cats:
        idx = [i for i, c in enumerate(categories) if c == cat]
        pts = coords[idx]
        hover = [
            f"<b>{labels[i]}</b><br>categoria: {cat}<br>"
            f"z=({pts[k,0]:.3f}, {pts[k,1]:.3f}"
            + (f", {pts[k,2]:.3f}" if dim == 3 else "") + ")"
            for k, i in enumerate(idx)
        ]
        common = dict(
            mode="markers",
            name=cat,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                size=marker_size,
                color=cat_color[cat],
                opacity=opacity,
                line=dict(width=0.6, color="white"),
            ),
        )
        if dim == 3:
            fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], **common))
        else:
            fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], **common))

    # Elipsoides de covariância por categoria (dim=3 apenas)
    if dim == 3 and show_ellipsoids and len(unique_cats) > 1:
        import numpy as np

        # Bounding box global: escala os elipsoides para no máximo 35% do range
        # médio dos dados, garantindo que ambos fiquem visíveis independente
        # da dispersão absoluta de cada cluster.
        bbox = float(np.ptp(coords, axis=0).mean()) or 1.0
        max_radius = bbox * 0.35

        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 16)
        x_s = np.outer(np.cos(u), np.sin(v))
        y_s = np.outer(np.sin(u), np.sin(v))
        z_s = np.outer(np.ones_like(u), np.cos(v))
        sphere = np.stack([x_s.ravel(), y_s.ravel(), z_s.ravel()], axis=1)

        for cat in unique_cats:
            idx = [i for i, c in enumerate(categories) if c == cat]
            if len(idx) < 4:
                continue
            pts = coords[idx]
            mu = pts.mean(axis=0)
            cov = np.cov(pts.T)
            vals, vecs = np.linalg.eigh(cov)
            vals = np.maximum(vals, 0)
            radii = np.sqrt(vals)

            # Normaliza os semi-eixos para que o maior não exceda max_radius.
            # Isso garante que elipsoides de clusters dispersos (grande covariância)
            # não se estendam além dos limites visíveis do plot.
            largest = radii.max()
            if largest > max_radius:
                radii = radii * (max_radius / largest)

            ellipsoid = (sphere * radii) @ vecs.T + mu
            ex = ellipsoid[:, 0].reshape(x_s.shape)
            ey = ellipsoid[:, 1].reshape(y_s.shape)
            ez = ellipsoid[:, 2].reshape(z_s.shape)

            fig.add_trace(go.Surface(
                x=ex, y=ey, z=ez,
                colorscale=[[0, cat_color[cat]], [1, cat_color[cat]]],
                opacity=0.10,
                showscale=False,
                name=f"{cat} (elipsoide)",
                hoverinfo="skip",
                showlegend=False,
            ))

    axis_kw = dict(
        showgrid=True, gridcolor="rgba(200,200,200,0.4)",
        showbackground=True, backgroundcolor="rgba(240,240,245,0.6)",
    )
    if dim == 3:
        fig.update_layout(
            title=dict(text=title_full, x=0.5, xanchor="center"),
            scene=dict(
                xaxis=dict(title="z₁", **axis_kw),
                yaxis=dict(title="z₂", **axis_kw),
                zaxis=dict(title="z₃", **axis_kw),
            ),
            legend=dict(title="Categoria", borderwidth=1),
            margin=dict(l=0, r=0, t=60, b=0),
            width=width, height=height,
        )
    else:
        fig.update_layout(
            title=dict(text=title_full, x=0.5, xanchor="center"),
            xaxis_title="z₁", yaxis_title="z₂",
            legend=dict(title="Categoria", borderwidth=1),
            width=width, height=height,
        )

    return fig


def semantic_overlap_score(
    K: torch.Tensor,
    categories: list[str],
    center: bool = True,
    dim: int = 3,
) -> dict[tuple[str, str], float]:
    """Pontuação de sobreposição semântica entre pares de categorias.

    Para cada par (cat_a, cat_b), computa a distância média entre os centroides
    das duas categorias no espaço espectral de ``dim`` dimensões, normalizada
    pela raiz da soma dos raios médios (desvio padrão dos clusters). Valores
    próximos de 0 indicam forte sobreposição; valores > 1 indicam separação.

    Retorna um dicionário ``{(cat_a, cat_b): score}`` para todos os pares.
    """
    from ..gram.eigenspace import spectral_coordinates
    import numpy as np

    K_work = K.detach().float()
    K_work = 0.5 * (K_work + K_work.T)
    if center:
        K_work = _center(K_work)

    coords = spectral_coordinates(K_work, dim=dim).cpu().numpy()
    unique_cats = list(dict.fromkeys(categories))

    centroids: dict[str, np.ndarray] = {}
    radii: dict[str, float] = {}
    for cat in unique_cats:
        idx = [i for i, c in enumerate(categories) if c == cat]
        pts = coords[idx]
        mu = pts.mean(axis=0)
        centroids[cat] = mu
        radii[cat] = float(np.linalg.norm(pts - mu, axis=1).mean()) if len(idx) > 1 else 0.0

    scores: dict[tuple[str, str], float] = {}
    for i, ca in enumerate(unique_cats):
        for cb in unique_cats[i + 1:]:
            dist = float(np.linalg.norm(centroids[ca] - centroids[cb]))
            scale = (radii[ca] + radii[cb]) / 2 + 1e-12
            scores[(ca, cb)] = dist / scale
    return scores

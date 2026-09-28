"""Identidade visual das figuras do artigo, sobre seaborn.

A base é `seaborn.set_theme`: contexto `paper`, estilo `ticks`, paleta `colorblind`. É
o que padroniza -- todo módulo de figura chama `T.use()` e nenhum ajusta estilo por
conta própria. O que este módulo acrescenta ao tema do seaborn é só aquilo sobre o que
ele não tem opinião: tipografia casada com o LaTeX, larguras de coluna, separador
decimal em vírgula e gravação em PDF.

Tipografia
----------
Não há LaTeX no PATH, então `text.usetex` está fora. A combinação disponível que casa
com um artigo é **Nimbus Roman** (clone métrico do Times) para o texto e **STIX** para
a matemática -- o par usado por várias editoras, e o STIX vem embutido no matplotlib,
de modo que a figura não depende de fonte instalada. Os corpos são absolutos em pt e
dimensionados para a figura entrar na página **sem reescala**: reescalar no LaTeX é o
que produz aquele artigo em que cada figura tem um corpo diferente.

Paleta
------
`colorblind` do seaborn, que é a paleta publicada de Okabe--Ito. Duas fatias ficam fora
das séries: o cinza (índice 7) é sempre o controle, e o rosa-claro (índice 6) tem
contraste baixo demais contra o papel. Além do matiz, **toda série tem marcador
próprio**, fixo em todas as figuras -- o que mantém as figuras legíveis impressas em
escala de cinza e cobre o caso de daltonismo severo.

Geometria
---------
Larguras em polegadas, medidas da caixa de texto e não do papel:

  col   3,35"  uma coluna de artigo em duas colunas
  wide  5,00"  figura destacada em texto de coluna única
  full  6,90"  largura total da caixa de texto
  page  9,20"  figura em página apaisada (as grades de 7 blocos)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------

# Ordem canônica das séries. As cores saem da paleta padrão do seaborn nesta ordem --
# nenhuma fatia é escolhida à mão. A ordem é fixa para que uma série tenha a mesma cor em
# todas as figuras, mesmo quando uma figura mostra só um subconjunto delas.
ORDER = ["joint", "joint_root", "params", "activations", "grads",
         "rsa_euclid", "joint_ag", "rsa_corr"]
LABEL = {
    "joint": r"conjunta $(W,A,\Gamma)$",
    "joint_root": r"conjunta, raiz $d$-ésima",
    "params": r"só $W$ (parâmetros)",
    "activations": r"só $A$ (ativações)",
    "grads": r"só $\Gamma$ (gradiente)",
    "rsa_euclid": "RSA euclidiana",
    "joint_ag": r"conjunta $(A,\Gamma)$",
    "rsa_corr": r"RSA $1-$Pearson",
}
PALETTE = sns.color_palette("colorblind", n_colors=len(ORDER))
MARKERS = ["o", "s", "^", "D", "v", "X", "P", "*"]

COLOR = dict(zip(ORDER, PALETTE))
MARKER = dict(zip(ORDER, MARKERS))
SERIES = {k: (COLOR[k], MARKER[k], LABEL[k]) for k in ORDER}

CONTROL = "0.55"  # a rede aleatória: neutra e tracejada, nunca uma cor da paleta
INK = "black"
INK_SOFT = "0.30"
INK_FAINT = "0.45"
RULE = "0.80"
SURFACE = "white"

SEQ = sns.color_palette("Blues", as_cmap=True)

# classe: as dez classes usam a mesma paleta, com o marcador separando os pares que
# compartilham matiz. Pares confundíveis (cat/dog, automobile/truck, deer/horse) nunca
# ficam no mesmo matiz.
CLASS_PAIRS = [(0, 8), (1, 9), (2, 6), (3, 7), (5, 4)]
CLASS_STYLE = {
    cls: (PALETTE[i], "o" if j == 0 else "^")
    for i, pair in enumerate(CLASS_PAIRS)
    for j, cls in enumerate(pair)
}

WIDTH = {"col": 3.35, "wide": 5.0, "full": 6.9, "page": 9.2}

# Título e nota de rodapé vão no `\caption` do LaTeX, não dentro da imagem. `T.title` e
# `T.note` viram no-ops; ligar esta chave devolve o texto para dentro da figura.
CAPTIONS_IN_FIGURE = False


def series(key: str) -> dict:
    """kwargs de plot para uma série: cor e marcador vêm da ordem canônica."""
    color, marker, label = SERIES.get(key, (INK_SOFT, "o", key))
    return {"color": color, "marker": marker, "label": label, "markeredgewidth": 0.0}


def control(label: str = "controle", marker: str = "s") -> dict:
    """A rede aleatória: neutra, tracejada, marcador vazado.

    O marcador distingue os **dois** controles, que não são a mesma coisa: o cruzado
    (geometria aleatória contra o comportamento da rede treinada) e o auto-controle
    (cada rede contra o seu próprio comportamento).
    """
    return {
        "color": CONTROL, "marker": marker, "label": label, "linestyle": (0, (3.5, 2.5)),
        "markerfacecolor": SURFACE, "markeredgecolor": CONTROL, "markeredgewidth": 0.8,
    }


def hue_map(keys) -> dict:
    """rótulo -> cor, para passar em `palette=` das funções do seaborn."""
    return {SERIES[k][2]: SERIES[k][0] for k in keys}


def marker_map(keys) -> dict:
    """rótulo -> marcador, para passar em `markers=` das funções do seaborn."""
    return {SERIES[k][2]: SERIES[k][1] for k in keys}


# --------------------------------------------------------------------------
# tema
# --------------------------------------------------------------------------


def use() -> None:
    """Aplica o tema. Único ponto do projeto onde estilo é definido.

    O estilo é o do seaborn, sem reescrita: `paper` + `ticks` + `colorblind`. O `rc`
    abaixo não redefine estética -- cobre só o que o seaborn não decide e o artigo exige:
    a família tipográfica que casa com o LaTeX, a matemática em STIX (que vem embutida no
    matplotlib, então a figura não depende de fonte instalada) e o formato de saída, com
    fontes embutidas como TrueType, exigência comum de editora.
    """
    sns.set_theme(
        context="paper",
        style="ticks",
        palette="colorblind",
        font="serif",
        font_scale=0.85,
        rc={
            "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "figure.dpi": 200,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def grid(
    nrows: int = 1, ncols: int = 1, width: str | float = "full", height: float | None = None,
    ratio: float = 0.62, **kwargs,
):
    """Figura dimensionada pela largura da caixa de texto, não por tentativa e erro."""
    w = WIDTH.get(width, width) if isinstance(width, str) else width
    h = height if height is not None else w * ratio
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), squeeze=False, **kwargs)
    return fig, axes


def despine(ax, **kwargs) -> None:
    sns.despine(ax=ax, **kwargs)


def bare(ax) -> None:
    """Painel sem eixos: matrizes e dispersogramas, onde a escala não é lida."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(RULE)


def _comma(decimals: int):
    def fn(value, _pos):
        return f"{value:.{decimals}f}".replace(".", ",")
    return FuncFormatter(fn)


def decimal(ax, axis: str = "y", decimals: int = 2) -> None:
    """Separador decimal em vírgula: o artigo é em português."""
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(_comma(decimals))
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(_comma(decimals))


def panel_tags(
    axes, tags: list[str] | None = None, inside: bool = False,
    dx: float | None = None, dy: float | None = None,
) -> None:
    """(a), (b), (c)… para o texto poder citar cada painel.

    `inside` põe a etiqueta dentro da caixa: é o que se usa quando os títulos são
    longos, porque fora dela a etiqueta briga com o título e com o rótulo do eixo y.
    """
    flat = [a for a in (axes.ravel() if hasattr(axes, "ravel") else axes) if a.get_visible()]
    tags = tags or [f"({chr(97 + i)})" for i in range(len(flat))]
    for ax, tag in zip(flat, tags):
        if inside:
            ax.text(
                dx if dx is not None else 0.03, dy if dy is not None else 0.965, tag,
                transform=ax.transAxes, ha="left", va="top", fontsize=8.0, color=INK,
                fontweight="bold",
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0, alpha=0.85),
            )
        else:
            ax.text(
                dx if dx is not None else -0.02, dy if dy is not None else 1.06, tag,
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                color=INK, fontweight="bold",
            )


def title(fig, text: str, y: float = 1.0) -> None:
    """Título -- só aparece com `CAPTIONS_IN_FIGURE`; o normal é ir no LaTeX."""
    if not CAPTIONS_IN_FIGURE:
        return
    fig.suptitle(text, y=y)


def note(fig, text: str, y: float = 0.0, width: int | None = None) -> None:
    """Nota de rodapé -- idem; o `\\caption` do LaTeX é o lugar dela."""
    if not CAPTIONS_IN_FIGURE:
        return
    import textwrap
    if width:
        text = textwrap.fill(text, width=width)
    fig.text(0.5, y, text, ha="center", va="top", fontsize=7.0, color=INK_FAINT,
             linespacing=1.5)


def class_legend(fig, classes: list[str], ncol: int = 5, y: float = 0.0) -> None:
    handles = [
        plt.Line2D(
            [], [], linestyle="none", marker=CLASS_STYLE[i][1], markersize=4.2,
            markerfacecolor=CLASS_STYLE[i][0], markeredgecolor=SURFACE,
            markeredgewidth=0.4, label=name,
        )
        for i, name in enumerate(classes)
    ]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, y))


def save(fig, path: str | Path, formats: tuple[str, ...] = ("pdf", "png")) -> None:
    """Grava vetor para o LaTeX e bitmap para inspeção, com o mesmo nome base."""
    base = Path(path).with_suffix("")
    for ext in formats:
        fig.savefig(base.with_suffix(f".{ext}"))
    plt.close(fig)

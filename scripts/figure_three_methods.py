"""Figura "três leituras do mesmo par de clientes" (README e documentação).

    python scripts/figure_three_methods.py                         # en -> docs/lib/assets/images/
    python scripts/figure_three_methods.py --lang pt --out fig.webp  # outra língua, outro destino

Sem argumentos, grava docs/lib/assets/images/credito_tres_metodos.webp (inglês, a versão do README e
da documentação). A versão em português é gerada só quando pedida, no caminho dado por --out.
As caixas têm a largura do texto e as linhas vão de borda a borda, então o layout se ajusta ao idioma.
O estilo de cada caixa vem de quantos métodos apontam o atributo (três, dois ou um).
"""

from __future__ import annotations

from collections import Counter
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W, H, DPI = 2058, 1283, 200          # pixels da imagem final; 1 unidade dos eixos = 1 pixel
FONT = "Open Sans"

# as cinco primeiras posições de cada método (HELOC, o mesmo par de clientes)
RANKINGS = {
    "Fannar": ["since_inquiry", "revolving", "inquiries", "account_age", "oldest_account"],
    "SHAP": ["inquiries", "since_inquiry", "revolving", "account_age", "inquiries_excl"],
    "LIME": ["inquiries", "revolving", "inquiries_excl", "account_age", "high_util_cards"],
}

TEXT = {
    "en": {
        "title": "Three readings of the same pair of clients",
        "subtitle": "The features that most separate A from B according to each method. "
                    "Where they agree, the conclusion is firmer.",
        "methods": {"Fannar": "distance in the model's space", "SHAP": "difference in contributions",
                    "LIME": "local linear model"},
        "features": {
            "since_inquiry": "Months since last inquiry",
            "revolving": "Revolving credit utilization (%)",
            "inquiries": "Inquiries, last 6 months",
            "account_age": "Average account age (months)",
            "oldest_account": "Months since oldest account",
            "inquiries_excl": "Inquiries, 6 months (excl. 7 days)",
            "high_util_cards": "Cards with high utilization",
        },
        "legend": ["pointed out by all three methods", "pointed out by two methods", "pointed out by one method"],
        "footer": "HELOC dataset (FICO). Model trained from scratch; values on the original scale.",
    },
    "pt": {
        "title": "Três leituras do mesmo par de clientes",
        "subtitle": "Os atributos que mais separam A de B segundo cada método. "
                    "Onde eles concordam, a conclusão fica mais firme.",
        "methods": {"Fannar": "distância no espaço do modelo", "SHAP": "diferença de contribuições",
                    "LIME": "modelo linear local"},
        "features": {
            "since_inquiry": "Meses desde a última consulta",
            "revolving": "Uso do crédito rotativo (%)",
            "inquiries": "Consultas em 6 meses",
            "account_age": "Idade média das contas (meses)",
            "oldest_account": "Meses desde a conta mais antiga",
            "inquiries_excl": "Consultas em 6 meses (exc. 7 dias)",
            "high_util_cards": "Cartões com uso alto",
        },
        "legend": ["apontado pelos três métodos", "apontado por dois métodos", "apontado por um método"],
        "footer": "Base HELOC (FICO). Modelo treinado do zero; valores na escala original.",
    },
}

INK, MUTED, FAINT = "#2b2f36", "#5b6472", "#6b7280"
BLUE, BLUE_SOFT, GREY = "#5d7dab", "#8fa7c7", "#d3d8df"
STYLE = {  # número de métodos -> (preenchimento, borda, texto, peso)
    3: (BLUE, BLUE, "white", "semibold"),
    2: ("white", BLUE_SOFT, INK, "normal"),
    1: ("white", GREY, INK, "normal"),
}
COLUMNS = {"Fannar": 480, "SHAP": 1060, "LIME": 1640}   # centros das colunas
ROW0, ROW_STEP, BOX_H, PAD_X, GAP = 403, 142, 62, 17, 16


def px(size: float) -> float:
    """Tamanho de fonte em pixels da imagem final -> pontos."""
    return size * 72 / DPI


DEFAULT_OUT = ROOT / "docs/lib/assets/images/credito_tres_metodos.webp"


def draw(lang: str, out: Path) -> None:
    t = TEXT[lang]
    votes = Counter(f for ranking in RANKINGS.values() for f in ranking)
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    fig.patch.set_facecolor("white")
    renderer = fig.canvas.get_renderer()

    ax.text(50, 75, t["title"], fontsize=px(46), fontweight="bold", color=INK, family=FONT, va="center")
    ax.text(50, 131, t["subtitle"], fontsize=px(28), color=MUTED, family=FONT, va="center")

    edges: dict[tuple[str, str], tuple[float, float, float]] = {}   # (método, atributo) -> (esq, dir, y)
    for method, cx in COLUMNS.items():
        ax.text(cx, 283, method, fontsize=px(38), fontweight="bold", color=INK, family=FONT, ha="center", va="center")
        ax.text(cx, 332, t["methods"][method], fontsize=px(25), color=MUTED, family=FONT, ha="center", va="center")
        for r, feature in enumerate(RANKINGS[method]):
            y = ROW0 + r * ROW_STEP
            fill, edge, color, weight = STYLE[votes[feature]]
            label = ax.text(cx, y + 1, t["features"][feature], fontsize=px(27), color=color, fontweight=weight,
                            family=FONT, ha="center", va="center", zorder=4)
            half = label.get_window_extent(renderer).width / 2 + PAD_X
            ax.add_patch(FancyBboxPatch((cx - half, y - BOX_H / 2), 2 * half, BOX_H,
                                        boxstyle="round,pad=0,rounding_size=15", facecolor=fill, edgecolor=edge,
                                        linewidth=px(3.5), zorder=3))
            edges[method, feature] = (cx - half, cx + half, y)

    methods = list(COLUMNS)
    for left, right in zip(methods, methods[1:]):
        for feature in RANKINGS[left]:
            if (right, feature) not in edges:
                continue
            _, x0, y0 = edges[left, feature]
            x1, _, y1 = edges[right, feature]
            strong = votes[feature] == 3
            ax.plot([x0 + GAP, x1 - GAP], [y0, y1], color=BLUE if strong else BLUE_SOFT,
                    linewidth=px(6 if strong else 3.5), solid_capstyle="round", zorder=2)

    x = 50
    for n, text in zip((3, 2, 1), t["legend"]):
        fill, edge, _, _ = STYLE[n]
        ax.add_patch(FancyBboxPatch((x, 1112), 24, 24, boxstyle="square,pad=0", facecolor=fill, edgecolor=edge,
                                    linewidth=px(3)))
        label = ax.text(x + 41, 1124, text, fontsize=px(27), color=MUTED, family=FONT, va="center")
        x += 41 + label.get_window_extent(renderer).width + 80
    ax.text(50, 1185, t["footer"], fontsize=px(23), color=FAINT, family=FONT, va="center")

    buffer = BytesIO()
    fig.savefig(buffer, dpi=DPI, format="png", facecolor="white")
    plt.close(fig)
    Image.open(buffer).convert("RGB").save(out, "WEBP", quality=90, method=6)
    print("gravado:", out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", choices=sorted(TEXT), default="en")
    parser.add_argument("--out", type=Path, default=None,
                        help="destino do .webp (padrão: docs/lib/assets/images/, só para --lang en)")
    args = parser.parse_args()
    if args.out is None and args.lang != "en":
        parser.error("--out é obrigatório para línguas além de en")
    draw(args.lang, args.out or DEFAULT_OUT)

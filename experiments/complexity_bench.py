"""Verificacao empirica da analise assintotica do metodo.

Tres varreduras, todas em CPU (numpy/torch, float64), com o menor tempo de `--repeats` execucoes:

  n      numero de objetos, com a dimensao das fontes fixa: Gram bruta por fonte, transformacoes
         por componente e Hadamard, centramento (como implementado, H G H, e por medias de linhas
         e colunas), distancias, k-vizinhos (ordenacao completa, como implementado, e selecao
         parcial), razao de participacao pelo traco e pela norma de Frobenius, e decomposicao
         espectral completa;
  D      dimensao de cada fonte, com n fixo: Gram bruta;
  F      numero de atributos na leitura de pares, num MLP residual nao treinado (o tempo nao
         depende do treinamento), para todas as tecnicas comparadas em P6, com as mesmas
         configuracoes dos estudos de caso:
           metodo             F + 2 linhas, uma passagem direta e uma reversa;
           Shapley exato      2^F avaliacoes (ate `--shapley-max-f`);
           Shapley amostrado  3 permutacoes, 3 (F + 1) avaliacoes;
           grad. integrados   64 passos, 64 passagens diretas e reversas;
           grad x delta x     uma passagem direta e uma reversa;
           LIME contrastivo   5 000 perturbacoes e uma regressao ponderada (ate `--lime-max-f`);
           SHAP (biblioteca)  PermutationExplainer, fundo de 100 amostras, max(1000, 2F + 1)
                              avaliacoes por amostra, para i e j (ate `--shap-max-f`);
           DiCE               metodo `random`, 5 contrafactuais (ate `--dice-max-f`).
         As tecnicas mais caras tem um F maximo para que a varredura termine em tempo razoavel;
         o limite vai no JSON e na figura.

O expoente de cada etapa e a inclinacao da reta de log(tempo) contra log(tamanho), ajustada nos
tamanhos maiores, onde os custos fixos pesam menos.

Uso:
    python complexity_bench.py
    python complexity_bench.py --only-f      # refaz so a varredura em F
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from letter_case_study import baseline_shapley, library_shap, lime_contrastive
from letter_budget import sampled_shapley
from src.attributions import build_dice, dice_scores, integrated_gradients
from src.mlp import ResidualMLP
from tabular_differences import gradxdelta, method_scores

# (chave, rotulo, serie do tema ou indice da paleta `colorblind`, marcador), na ordem da legenda
F_SERIES = (
    ("method", "método", "joint", None),
    ("shapley_exact", "Shapley exato", "params", None),
    ("shapley_perm3", "Shapley, 3 permutações", "grads", None),
    ("ig", "gradientes integrados (64 passos)", 9, "P"),
    ("gradxdelta", r"$\nabla f\cdot\Delta x$", "rsa_euclid", None),
    ("lime", "LIME contrastivo (5 000 perturbações)", "joint_root", None),
    ("shap", "SHAP da biblioteca (fundo de 100)", "activations", None),
    ("dice", "DiCE (5 contrafactuais)", 8, "*"),
)


def best_time(fn, repeats: int) -> float:
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def component(u: np.ndarray) -> np.ndarray:
    g = u @ u.T
    s = np.sqrt(np.maximum(np.diag(g), 1e-300))
    return (g / np.outer(s, s) + 1.0) / 2.0


def transform_and_compose(raws: list[np.ndarray]) -> np.ndarray:
    """T_a, deslocamento (+J)/2 e produto de Hadamard sobre as Grams brutas ja calculadas."""
    out = None
    for g in raws:
        s = np.sqrt(np.maximum(np.diag(g), 1e-300))
        c = (g / np.outer(s, s) + 1.0) / 2.0
        out = c if out is None else out * c
    return out


def center_hgh(g: np.ndarray) -> np.ndarray:
    n = len(g)
    h = np.eye(n) - 1.0 / n
    return h @ g @ h


def center_means(g: np.ndarray) -> np.ndarray:
    r = g.mean(1, keepdims=True)
    return g - r - r.T + g.mean()


def distances(g: np.ndarray) -> np.ndarray:
    dg = np.diag(g)
    return np.sqrt(np.maximum(dg[:, None] + dg[None, :] - 2 * g, 0))


def knn_sort(d: np.ndarray, k: int = 5) -> np.ndarray:
    dd = d + np.diag(np.full(len(d), np.inf))
    return np.argsort(dd, axis=1)[:, :k]


def knn_select(d: np.ndarray, k: int = 5) -> np.ndarray:
    dd = d + np.diag(np.full(len(d), np.inf))
    return np.argpartition(dd, k, axis=1)[:, :k]


def participation(g: np.ndarray) -> float:
    return float(np.trace(g) ** 2 / np.sum(g * g))


def slope(sizes, times, last: int = 3) -> float:
    x, y = np.log(np.asarray(sizes[-last:], float)), np.log(np.asarray(times[-last:], float))
    return float(np.polyfit(x, y, 1)[0])


def sweep_n(ns, dim: int, repeats: int, rng) -> dict:
    out = {k: [] for k in ("gram", "components", "center_hgh", "center_means", "distances",
                           "knn_sort", "knn_select", "participation", "eigh")}
    for n in ns:
        srcs = [rng.standard_normal((n, dim)) for _ in range(3)]
        out["gram"].append(best_time(lambda: [u @ u.T for u in srcs], repeats))
        raws = [u @ u.T for u in srcs]
        out["components"].append(best_time(lambda: transform_and_compose(raws), repeats))
        g = transform_and_compose(raws)
        out["center_hgh"].append(best_time(lambda: center_hgh(g), repeats))
        out["center_means"].append(best_time(lambda: center_means(g), repeats))
        gc = center_means(g)
        gc /= np.trace(gc)
        out["distances"].append(best_time(lambda: distances(gc), repeats))
        d = distances(gc)
        out["knn_sort"].append(best_time(lambda: knn_sort(d), repeats))
        out["knn_select"].append(best_time(lambda: knn_select(d), repeats))
        out["participation"].append(best_time(lambda: participation(gc), repeats))
        out["eigh"].append(best_time(lambda: np.linalg.eigh(gc), repeats))
        print(f"n = {n}: " + ", ".join(f"{k} {v[-1] * 1e3:.1f} ms" for k, v in out.items()), flush=True)
    return out


def sweep_d(dims, n: int, repeats: int, rng) -> list[float]:
    ts = []
    for dim in dims:
        u = rng.standard_normal((n, dim))
        ts.append(best_time(lambda: u @ u.T, repeats))
        print(f"D = {dim}: Gram {ts[-1] * 1e3:.1f} ms", flush=True)
    return ts


def sweep_f(fs, repeats: int, limits: dict) -> dict:
    """Tempo por par de cada tecnica; `out[k]` e alinhado com `out[f"fs_{k}"]`."""
    from lime.lime_tabular import LimeTabularExplainer

    device = torch.device("cpu")
    keys = [k for k, *_ in F_SERIES]
    out = {k: [] for k in keys} | {f"fs_{k}": [] for k in keys}
    rng = np.random.default_rng(0)
    for F in fs:
        torch.manual_seed(0)
        model = ResidualMLP(F, 10, 256, 6).eval()
        xi, xj = rng.standard_normal(F).astype(np.float32), rng.standard_normal(F).astype(np.float32)
        train = rng.standard_normal((500, F)).astype(np.float32)
        x64i, x64j = xi.astype(np.float64), xj.astype(np.float64)
        runs = {
            "method": (lambda: method_scores(model, xi, xj, device, 0, 1.0), repeats),
            "shapley_perm3": (lambda: sampled_shapley(model, x64i, x64j, 1, 2, device, 3 * (F + 1),
                                                      np.random.default_rng(0)), repeats),
            "ig": (lambda: integrated_gradients(model, xi, xj, 1, 2, device, 64), repeats),
            "gradxdelta": (lambda: gradxdelta(model, xi, xj, 1, 2, device), repeats),
        }
        if F <= limits["shapley_exact"]:
            runs["shapley_exact"] = (lambda: baseline_shapley(model, xi, xj, 1, 2, device), 2)
        if F <= limits["lime"]:
            lime = LimeTabularExplainer(train, mode="regression", discretize_continuous=False,
                                        sample_around_instance=True, random_state=0)
            runs["lime"] = (lambda: lime_contrastive(lime, model, x64i, x64j, 1, 2, device, 5000), 1)
        if F <= limits["shap"]:
            both = np.stack([xi, xj])
            runs["shap"] = (lambda: library_shap(model, both, train[:100], device, max(1000, 2 * F + 1), 0), 1)
        if F <= limits["dice"]:
            with torch.no_grad():
                labels = model(torch.from_numpy(train)).argmax(1).numpy()
                cj = int(model(torch.from_numpy(xj[None])).argmax())
            dice = build_dice(model, train, labels, device, 0)
            # a busca aleatoria do DiCE as vezes demora a achar um contrafactual (dezenas de segundos
            # numa unica chamada): o minimo de tres execucoes mede o custo tipico, como nas demais
            runs["dice"] = (lambda: dice_scores(dice, x64i, cj, 5, np.random.default_rng(0)), 3)
        for k, (fn, rep) in runs.items():
            fn()  # aquecimento: a primeira chamada paga alocacoes
            out[k].append(best_time(fn, rep))
            out[f"fs_{k}"].append(F)
        print(f"F = {F}: " + ", ".join(f"{k} {out[k][-1] * 1e3:.1f} ms" for k in runs), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ns", type=int, nargs="+", default=[500, 1000, 2000, 4000, 8000])
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--dims", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--n-for-d", type=int, default=1000)
    ap.add_argument("--fs", type=int, nargs="+", default=[8, 10, 12, 14, 16, 64, 256, 1024, 4096])
    ap.add_argument("--shapley-max-f", type=int, default=16)
    ap.add_argument("--lime-max-f", type=int, default=1024)
    ap.add_argument("--shap-max-f", type=int, default=256)
    ap.add_argument("--dice-max-f", type=int, default=64)
    ap.add_argument("--only-f", action="store_true", help="refaz so a varredura em F, reaproveitando n e D do JSON")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="outputs/complexity")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    torch.set_grad_enabled(True)

    limits = {"shapley_exact": args.shapley_max_f, "lime": args.lime_max_f, "shap": args.shap_max_f,
              "dice": args.dice_max_f}
    if args.only_f and (out / "results.json").exists():
        old = json.loads((out / "results.json").read_text())
        res_n, res_d = old["n"], old["D"]
    else:
        res_n = sweep_n(args.ns, args.dim, args.repeats, rng)
        res_d = sweep_d(args.dims, args.n_for_d, args.repeats, rng)
    res_f = sweep_f(args.fs, args.repeats, limits)
    # expoente em F nos tres maiores F de cada tecnica; o Shapley exato e exponencial e vai como
    # log2(tempo) por atributo, que deve ficar perto de 1 (o tempo dobra a cada atributo)
    f_slopes = {k: slope(res_f[f"fs_{k}"], res_f[k]) for k, *_ in F_SERIES
                if k != "shapley_exact" and len(res_f[k]) >= 3}
    f_slopes["shapley_exact_log2_per_feature"] = float(
        np.polyfit(res_f["fs_shapley_exact"][-3:], np.log2(res_f["shapley_exact"][-3:]), 1)[0])
    slopes = {"n": {k: slope(args.ns, v) for k, v in res_n.items()},
              "D": {"gram": slope(args.dims, res_d)}, "F": f_slopes}
    (out / "results.json").write_text(json.dumps({"args": vars(args), "n": res_n, "D": res_d, "F": res_f,
                                                  "slopes": slopes}, indent=2))
    print(json.dumps(slopes, indent=2))
    plot(args, res_n, res_d, res_f, slopes, str(out / "complexity.png"))


def _comma(v: float) -> str:
    return f"{v:.2f}".replace(".", ",")


def _below(ax, fontsize: float = 5.6) -> None:
    """Legenda fora do painel, logo abaixo do rotulo do eixo x."""
    ax.legend(fontsize=fontsize, loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False,
              handlelength=1.8, borderaxespad=0.0)


def plot(args, res_n, res_d, res_f, slopes, path: str) -> None:
    """(a) etapas contra n, (b) Gram contra D, (c) leitura de pares contra F; log-log.

    As legendas ficam abaixo dos paineis, fora da area dos dados; em (b), com uma serie so, o
    expoente vai como anotacao. Os eixos em n e em F tem marcas explicitas, sem a notacao 10^k
    sobreposta que o eixo logaritmico gera por padrao.
    """
    import seaborn as sns
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    from src import theme as T
    T.use()
    import matplotlib.pyplot as plt  # noqa: F401
    extra = sns.color_palette("colorblind", 10)
    fig, axes = T.grid(1, 3, width="full", height=3.6, gridspec_kw={"width_ratios": [1.0, 0.75, 1.25]})

    ax = axes[0, 0]
    ns = np.asarray(args.ns, float)
    names = {"gram": ("Gram bruta (3 fontes)", "params"), "components": ("transformações e Hadamard", "activations"),
             "distances": ("distâncias", "grads"), "center_means": ("centramento por médias", "rsa_euclid"),
             "center_hgh": ("centramento $HGH$", "joint_root"), "eigh": ("decomposição espectral", "joint")}
    for k, (lab, key) in names.items():
        ax.plot(ns, res_n[k], ms=3, lw=1.2, **{**T.series(key), "label": f"{lab} ({_comma(slopes['n'][k])})"})
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ns))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:,.0f}".replace(",", " ")))
    ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("objetos $n$"); ax.set_ylabel("tempo (s)")
    _below(ax)
    T.despine(ax)

    ax = axes[0, 1]
    ax.plot(args.dims, res_d, ms=3, lw=1.2, **T.series("params"))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.text(0.04, 0.95, f"Gram bruta\ninclinação {_comma(slopes['D']['gram'])}", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.0, color=T.INK_SOFT)
    ax.set_xlabel(f"dimensão da fonte $D$ ($n = {args.n_for_d}$)"); ax.set_ylabel("tempo (s)")
    T.despine(ax)

    ax = axes[0, 2]
    fs_all = args.fs
    for k, lab, key, marker in F_SERIES:
        if not res_f.get(k):
            continue
        style = T.series(key) if isinstance(key, str) else {"color": extra[key], "marker": marker,
                                                               "markeredgewidth": 0.0}
        xs = res_f[f"fs_{k}"]
        if k == "shapley_exact":
            tag = f"{lab} (dobra a cada atributo)"
        elif k in slopes["F"] and slopes["F"][k] < 0.2:
            # numero fixo de passagens pela rede (ou orcamento fixo, no SHAP): o custo nao depende de F
            tag = f"{lab} (constante em $F$)"
        elif k in slopes["F"]:
            tag = f"{lab} ({_comma(slopes['F'][k])})"
        else:
            tag = lab
        if xs[-1] < fs_all[-1]:
            tag += f", até $F = {xs[-1]}$"
        ax.plot(xs, res_f[k], ms=3, lw=1.2, **{**style, "label": tag})
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("atributos $F$"); ax.set_ylabel("tempo por par (s)")
    _below(ax)
    T.despine(ax)

    T.panel_tags(axes)
    fig.tight_layout()
    T.save(fig, path)


if __name__ == "__main__":
    main()

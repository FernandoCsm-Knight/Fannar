"""Verificacao empirica da analise assintotica do metodo.

Tres varreduras, todas em CPU (numpy/torch, float64), com o menor tempo de `--repeats` execucoes:

  n      numero de objetos, com a dimensao das fontes fixa: Gram bruta por fonte, transformacoes
         por componente e Hadamard, centramento (como implementado, H G H, e por medias de linhas
         e colunas), distancias, k-vizinhos (ordenacao completa, como implementado, e selecao
         parcial), razao de participacao pelo traco e pela norma de Frobenius, e decomposicao
         espectral completa;
  D      dimensao de cada fonte, com n fixo: Gram bruta;
  F      numero de atributos na leitura de pares, num MLP residual nao treinado (o tempo nao
         depende do treinamento): o metodo (F + 2 linhas, uma passagem direta e uma reversa),
         o Shapley exato (2^F avaliacoes) e o Shapley por permutacoes (m (F + 1) avaliacoes).

O expoente de cada etapa e a inclinacao da reta de log(tempo) contra log(tamanho), ajustada nos
tamanhos maiores, onde os custos fixos pesam menos.

Uso:
    python complexity_bench.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from letter_case_study import baseline_shapley
from letter_budget import sampled_shapley
from src.mlp import ResidualMLP
from tabular_differences import method_scores


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


def sweep_f(fs, repeats: int, shapley_max_f: int) -> dict:
    device = torch.device("cpu")
    out = {"method": [], "shapley_exact": [], "shapley_perm3": [], "fs_exact": []}
    rng = np.random.default_rng(0)
    for F in fs:
        torch.manual_seed(0)
        model = ResidualMLP(F, 10, 256, 6).eval()
        xi, xj = rng.standard_normal(F).astype(np.float32), rng.standard_normal(F).astype(np.float32)
        method_scores(model, xi, xj, device, 0, 1.0)  # aquecimento: a primeira chamada paga alocacoes
        out["method"].append(best_time(lambda: method_scores(model, xi, xj, device, 0, 1.0), repeats))
        out["shapley_perm3"].append(best_time(
            lambda: sampled_shapley(model, xi.astype(np.float64), xj.astype(np.float64), 1, 2, device,
                                    3 * (F + 1), np.random.default_rng(0)), repeats))
        if F <= shapley_max_f:
            baseline_shapley(model, xi, xj, 1, 2, device)
            out["shapley_exact"].append(best_time(lambda: baseline_shapley(model, xi, xj, 1, 2, device), 2))
            out["fs_exact"].append(F)
        print(f"F = {F}: método {out['method'][-1] * 1e3:.1f} ms, Shapley por 3 permutações "
              f"{out['shapley_perm3'][-1] * 1e3:.1f} ms"
              + (f", Shapley exato {out['shapley_exact'][-1] * 1e3:.1f} ms" if F <= shapley_max_f else ""),
              flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ns", type=int, nargs="+", default=[500, 1000, 2000, 4000, 8000])
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--dims", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--n-for-d", type=int, default=1000)
    ap.add_argument("--fs", type=int, nargs="+", default=[8, 10, 12, 14, 16, 64, 256, 1024, 4096])
    ap.add_argument("--shapley-max-f", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="outputs/complexity")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    torch.set_grad_enabled(True)

    res_n = sweep_n(args.ns, args.dim, args.repeats, rng)
    res_d = sweep_d(args.dims, args.n_for_d, args.repeats, rng)
    res_f = sweep_f(args.fs, args.repeats, args.shapley_max_f)
    slopes = {"n": {k: slope(args.ns, v) for k, v in res_n.items()},
              "D": {"gram": slope(args.dims, res_d)},
              "F": {"method": slope(args.fs, res_f["method"]),
                    "shapley_perm3": slope(args.fs, res_f["shapley_perm3"]),
                    "shapley_exact_log2_per_feature": float(np.polyfit(res_f["fs_exact"][-3:],
                                                                        np.log2(res_f["shapley_exact"][-3:]), 1)[0])}}
    (out / "results.json").write_text(json.dumps({"args": vars(args), "n": res_n, "D": res_d, "F": res_f,
                                                  "slopes": slopes}, indent=2))
    print(json.dumps(slopes, indent=2))
    plot(args, res_n, res_d, res_f, slopes, str(out / "complexity.png"))


def plot(args, res_n, res_d, res_f, slopes, path: str) -> None:
    """(a) etapas contra n, (b) Gram contra D, (c) leitura de pares contra F; log-log."""
    from src import theme as T
    T.use()
    import matplotlib.pyplot as plt  # noqa: F401
    fig, axes = T.grid(1, 3, width="full", height=2.5)
    ax = axes[0, 0]
    ns = np.asarray(args.ns, float)
    names = {"gram": ("Gram bruta (3 fontes)", "params"), "components": ("transformações e Hadamard", "activations"),
             "distances": ("distâncias", "grads"), "center_means": ("centramento por médias", "rsa_euclid"),
             "center_hgh": ("centramento $HGH$", "joint_root"), "eigh": ("decomposição espectral", "joint")}
    for k, (lab, key) in names.items():
        st = T.series(key)
        ax.plot(ns, res_n[k], ms=3, lw=1.2, **{**st, "label": f"{lab} ({slopes['n'][k]:.2f})".replace(".", ",")})
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("objetos $n$"); ax.set_ylabel("tempo (s)")
    ax.legend(fontsize=5.2, loc="upper left")
    T.despine(ax)
    ax = axes[0, 1]
    st = T.series("params")
    ax.plot(args.dims, res_d, ms=3, lw=1.2, **{**st, "label": f"Gram bruta ({slopes['D']['gram']:.2f})".replace(".", ",")})
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"dimensão da fonte $D$ ($n = {args.n_for_d}$)"); ax.set_ylabel("tempo (s)")
    ax.legend(fontsize=5.6, loc="upper left")
    T.despine(ax)
    ax = axes[0, 2]
    for k, key, lab, xs in (("method", "joint", f"método ({slopes['F']['method']:.2f})", args.fs),
                            ("shapley_perm3", "grads", f"Shapley, 3 permutações ({slopes['F']['shapley_perm3']:.2f})", args.fs),
                            ("shapley_exact", "params", "Shapley exato", res_f["fs_exact"])):
        st = T.series(key)
        ax.plot(xs, res_f[k], ms=3, lw=1.2, **{**st, "label": lab.replace(".", ",")})
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("atributos $F$"); ax.set_ylabel("tempo por par (s)")
    ax.legend(fontsize=5.6, loc="upper left")
    T.despine(ax)
    T.panel_tags(axes)
    fig.tight_layout()
    T.save(fig, path)


if __name__ == "__main__":
    main()

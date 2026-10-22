"""Datacao sem tendencia e testes estatisticos (itens (f) e (h) da Subsecao `subsec:adicionais`).

Nao treina nada: le os `runs/*.json` ja gravados pelas suites, pelas ablacoes e pela leitura de
pares, e escreve `outputs/stats/tables.md` e `outputs/stats/stats.json`.

(f) Datacao. A estabilidade a intervencao (1 − taxa de troca com σ = 2) e a recuperacao da
    predicao pela Gram conjunta crescem as duas com a profundidade, e o Spearman entre elas
    herda essa tendencia comum. Aqui ele e recalculado sobre as diferencas entre blocos
    consecutivos, que removem a tendencia: sobra so a pergunta "os blocos em que a decisao mais
    se estabiliza sao os blocos em que a geometria mais ganha?".

(h) Estatistica. Para cada afirmacao do tipo "a conjunta supera X" (as da suite e as das
    ablacoes), em cada bloco:
      - Wilcoxon pareado bilateral, com correcao de Holm sobre todas as afirmacoes e blocos de
        uma familia;
      - intervalo de confianca bootstrap (percentil, 95%) da diferenca media.
    A unidade e a base (media entre sementes) quando ha varias bases, como recomenda Demsar
    (2006), e a semente quando ha uma base so. Com n unidades, o menor p bilateral possivel do
    Wilcoxon e 2/2^n: com tres sementes (Pet) e 0,25, e nenhuma afirmacao pode ser significativa;
    a tabela informa esse piso em vez de omitir o teste.
    Entre mais de dois metodos sobre as bases tabulares: Friedman e o pos-teste de Nemenyi
    (diferenca critica dos postos medios, alfa = 0,05).

Uso:
    python ablations_stats.py
    python ablations_stats.py --suite tabular=outputs/tabular_suite --suite pets_vit=outputs/pets_vit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon

from src.ablation import COMPOSITIONS, LABEL as ALABEL, READINGS as AREADINGS, REPS
from src.suite import CLAIMS

SUITES = {"tabular": "outputs/tabular_suite", "pets_resnet": "outputs/pets_suite",
          "pets_vit": "outputs/pets_suite_vit", "cifar100_resnet": "outputs/cifar100_suite"}
ABLATIONS = {"tabular": "outputs/ablations/tabular", "pets_resnet": "outputs/ablations/pets_resnet",
             "pets_vit": "outputs/ablations/pets_vit", "cifar100_resnet": "outputs/ablations/cifar100_resnet"}
DIFFERENCES = "outputs/tabular_differences"

# valores criticos q_alfa do teste de Nemenyi para alfa = 0,05, k = 2..10 (Demsar, 2006, tabela 5)
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def load_runs(path: str | Path) -> list[dict]:
    files = sorted(Path(path, "runs").glob("*.json"))
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


# --------------------------------------------------------------------------
# (f) datacao
# --------------------------------------------------------------------------


def dating(runs: list[dict], variant: str = "fixed") -> dict:
    raw, detrended = [], []
    for r in runs:
        blocks = r["blocks"]
        stab = np.array([1.0 - r["flip"][f"{b}@2"] for b in blocks])
        curve = np.array([r["variants"][variant]["metrics"][b]["joint"]["knn_pred"] for b in blocks])
        if np.std(stab) > 0 and np.std(curve) > 0:
            raw.append(spearmanr(stab, curve).statistic)
        ds, dc = np.diff(stab), np.diff(curve)
        if np.std(ds) > 0 and np.std(dc) > 0:
            detrended.append(spearmanr(ds, dc).statistic)
    raw, detrended = np.asarray(raw, dtype=float), np.asarray(detrended, dtype=float)
    return {"n": len(runs), "raw": (float(np.nanmean(raw)), float(np.nanstd(raw))),
            "detrended": (float(np.nanmean(detrended)), float(np.nanstd(detrended))),
            "detrended_positive": float(np.mean(detrended > 0)) if len(detrended) else float("nan"),
            "detrended_wilcoxon_p": _wilcoxon(detrended)}


# --------------------------------------------------------------------------
# (h) testes pareados
# --------------------------------------------------------------------------


def _wilcoxon(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2 or np.all(diff == 0):
        return float("nan")
    return float(wilcoxon(diff, zero_method="zsplit", alternative="two-sided").pvalue)


def _bootstrap(diff: np.ndarray, reps: int, rng) -> tuple[float, float]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return float("nan"), float("nan")
    means = rng.choice(diff, size=(reps, len(diff)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        out[idx] = running
    return out.tolist()


def _units(runs: list[dict], get) -> np.ndarray:
    """Um valor por base (media entre sementes) ou por semente, se ha uma base so."""
    names = list(dict.fromkeys(r["dataset"] for r in runs))
    if len(names) == 1:
        return np.array([get(r) for r in runs], dtype=float)
    return np.array([np.nanmean([get(r) for r in runs if r["dataset"] == n]) for n in names], dtype=float)


def paired_tests(runs: list[dict], comparisons: list[tuple], blocks: list[str], reps: int, seed: int) -> list[dict]:
    """`comparisons`: (rotulo, funcao (run, bloco) -> (conjunta, alternativa))."""
    rng = np.random.default_rng(seed)
    rows = []
    for label, get in comparisons:
        for b in blocks:
            mine = _units(runs, lambda r: get(r, b)[0])
            other = _units(runs, lambda r: get(r, b)[1])
            diff = mine - other
            ok = np.isfinite(diff)
            rows.append({"claim": label, "block": b, "n": int(ok.sum()),
                         "mean_diff": float(np.nanmean(diff)) if ok.any() else float("nan"),
                         "ci": _bootstrap(diff, reps, rng), "p": _wilcoxon(diff),
                         "p_floor": 2.0 / 2 ** int(ok.sum()) if ok.any() else float("nan")})
    # Holm dentro de cada afirmacao, sobre os blocos em que ela e avaliada (a familia de hipoteses
    # de uma afirmacao como "a conjunta supera o piso"); corrigir sobre todas as afirmacoes juntas
    # tornaria qualquer teste com 10 bases insignificante por construcao
    for label in dict.fromkeys(r["claim"] for r in rows):
        group = [r for r in rows if r["claim"] == label]
        for row, adj in zip(group, holm([r["p"] for r in group])):
            row["p_holm"] = adj
    return rows


def suite_comparisons(variant: str = "fixed") -> list[tuple]:
    out = []
    for label, reading, against in CLAIMS:
        def get(r, b, reading=reading, against=against):
            v = r["variants"][variant]
            mine = v["metrics"][b]["joint"].get(reading, np.nan)
            if against == "floor":
                other = v["floor"][b].get(reading, np.nan)
            elif against.startswith("key:"):
                other = v["metrics"][b]["joint"].get(against[4:], np.nan)
            else:
                other = v["metrics"][b][against].get(reading, np.nan)
            return _num(mine), _num(other)
        out.append((label, get))
    return out


def ablation_comparisons() -> list[tuple]:
    out = []
    for reading in AREADINGS:
        for rep in REPS[1:]:
            out.append((f"{reading}: produto > {ALABEL[rep]}",
                        lambda r, b, reading=reading, rep=rep: (_num(r["metrics"][b]["joint"][reading]),
                                                                _num(r["metrics"][b][rep][reading]))))
        for rep in COMPOSITIONS:
            out.append((f"{reading}: {ALABEL[rep]} > piso",
                        lambda r, b, reading=reading, rep=rep: (_num(r["metrics"][b][rep][reading]),
                                                                _num(r["floor"][b][rep][reading]))))
    return out


def _num(v) -> float:
    return float("nan") if v is None else float(v)


# --------------------------------------------------------------------------
# Friedman e Nemenyi
# --------------------------------------------------------------------------


def friedman_nemenyi(table: np.ndarray, methods: list[str]) -> dict:
    """`table`: bases × metodos, maior e melhor. Postos medios (1 = melhor) e diferenca critica.

    Um metodo indefinido em todas as bases (Ω no ultimo bloco sob alvo fixo) sai do teste antes
    das bases; depois, sai toda base em que algum metodo restante e indefinido.
    """
    cols = np.any(np.isfinite(table), 0)
    table, methods = table[:, cols], [m for m, c in zip(methods, cols) if c]
    keep = np.all(np.isfinite(table), 1)
    t = table[keep]
    n, k = t.shape
    if n < 3 or k < 3:
        return {"n": int(n), "methods": methods, "p": float("nan"), "ranks": [float("nan")] * k, "cd": float("nan")}
    ranks = np.array([np.argsort(np.argsort(-row, kind="stable"), kind="stable") + 1 for row in t], dtype=float)
    # empates recebem o posto medio
    for i, row in enumerate(t):
        for v in np.unique(row):
            tie = row == v
            if tie.sum() > 1:
                ranks[i, tie] = ranks[i, tie].mean()
    p = float(friedmanchisquare(*t.T).pvalue)
    cd = NEMENYI_Q05.get(k, float("nan")) * np.sqrt(k * (k + 1) / (6.0 * n))
    return {"n": int(n), "methods": methods, "p": p, "ranks": ranks.mean(0).tolist(), "cd": float(cd)}


def ablation_friedman(runs: list[dict], reading: str, block: str) -> dict:
    names = list(dict.fromkeys(r["dataset"] for r in runs))
    table = np.array([[np.nanmean([_num(r["metrics"][block][rep][reading]) for r in runs if r["dataset"] == n])
                       for rep in REPS] for n in names])
    return friedman_nemenyi(table, [ALABEL[rep] for rep in REPS])


def differences_friedman(runs: list[dict], block: str) -> dict:
    keys = [f"method@{block}", "absdiff", "gradxdelta", "ig", "random", f"floor@{block}"]
    labels = ["método", "\\|Δx\\|", "grad×Δx", "grad. integrados", "aleatória", "piso"]
    keep = [k in runs[0]["auc"] for k in keys]
    keys, labels = [k for k, ok in zip(keys, keep) if ok], [m for m, ok in zip(labels, keep) if ok]
    names = list(dict.fromkeys(r["dataset"] for r in runs))
    table = np.array([[np.mean([r["auc"][k] for r in runs if r["dataset"] == n]) for k in keys] for n in names])
    return friedman_nemenyi(table, labels)


# --------------------------------------------------------------------------
# relatorio
# --------------------------------------------------------------------------


def _f(v, d=3) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.{d}f}".replace(".", ",")


def _p(v) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return "< 0,001" if v < 1e-3 else _f(v, 3)


def tests_table(rows: list[dict]) -> list[str]:
    L = ["| afirmação | bloco | n | diferença média | IC 95% | p | p (Holm) | menor p possível |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lo, hi = r["ci"]
        L.append(f"| {r['claim']} | {r['block']} | {r['n']} | {_f(r['mean_diff'])} | [{_f(lo)}; {_f(hi)}] | "
                 f"{_p(r['p'])} | {_p(r['p_holm'])} | {_p(r['p_floor'])} |")
    return L


def friedman_lines(title: str, res: dict) -> list[str]:
    L = [f"**{title}** — {res['n']} bases, Friedman p = {_p(res['p'])}, diferença crítica de Nemenyi "
         f"{_f(res['cd'], 2)}.\n", "| método | posto médio |", "|---|---|"]
    for m, rk in sorted(zip(res["methods"], res["ranks"]), key=lambda x: x[1]):
        L.append(f"| {m} | {_f(rk, 2)} |")
    return L + [""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", action="append", default=[], help="nome=pasta de uma suite")
    ap.add_argument("--ablation", action="append", default=[], help="nome=pasta de uma ablacao")
    ap.add_argument("--differences", default=DIFFERENCES)
    ap.add_argument("--friedman-blocks", nargs="+", default=["b1", "b3", "b6"])
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/stats")
    args = ap.parse_args()

    suites = SUITES | dict(s.split("=", 1) for s in args.suite)
    ablations = ABLATIONS | dict(s.split("=", 1) for s in args.ablation)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result: dict = {"dating": {}, "suite_tests": {}, "ablation_tests": {}, "friedman": {}}
    L = ["# Datação sem tendência e testes estatísticos\n"]

    L += ["## (f) Datação: Spearman entre estabilidade à intervenção e k-vizinhos da conjunta\n",
          "Por execução, alvo fixo. \"Bruto\" é o valor da Subseção P8; \"sem tendência\" usa as diferenças "
          "entre blocos consecutivos. O p é do Wilcoxon sobre os valores sem tendência contra zero.\n",
          "| família | execuções | bruto | sem tendência | fração > 0 | p |", "|---|---|---|---|---|---|"]
    for name, path in suites.items():
        runs = load_runs(path)
        if not runs:
            continue
        d = dating(runs)
        result["dating"][name] = d
        L.append(f"| {name} | {d['n']} | {_f(d['raw'][0], 2)} ± {_f(d['raw'][1], 2)} | "
                 f"{_f(d['detrended'][0], 2)} ± {_f(d['detrended'][1], 2)} | {_f(d['detrended_positive'], 2)} | "
                 f"{_p(d['detrended_wilcoxon_p'])} |")
    L.append("")

    L += ["## (h) Afirmações da suíte\n"]
    for name, path in suites.items():
        runs = load_runs(path)
        if not runs:
            continue
        rows = paired_tests(runs, suite_comparisons(), runs[0]["blocks"], args.bootstrap, args.seed)
        result["suite_tests"][name] = rows
        L += [f"### {name}\n"] + tests_table(rows) + [""]

    L += ["## (h) Ablações da composição\n"]
    for name, path in ablations.items():
        runs = load_runs(path)
        if not runs:
            continue
        rows = paired_tests(runs, ablation_comparisons(), runs[0]["blocks"], args.bootstrap, args.seed)
        result["ablation_tests"][name] = rows
        L += [f"### {name}\n"] + tests_table(rows) + [""]
        if len({r["dataset"] for r in runs}) >= 3:
            for reading in ("correctness_auc", "knn_pred"):
                for b in args.friedman_blocks:
                    if b not in runs[0]["blocks"]:
                        continue
                    res = ablation_friedman(runs, reading, b)
                    result["friedman"][f"{name}/{reading}/{b}"] = res
                    L += friedman_lines(f"{name}, {reading}, {b}", res)

    diff_runs = load_runs(args.differences)
    if diff_runs:
        L += ["## (h) Leitura de pares nas dez bases\n"]
        for b in args.friedman_blocks:
            if b in diff_runs[0]["blocks"]:
                res = differences_friedman(diff_runs, b)
                result["friedman"][f"differences/{b}"] = res
                L += friedman_lines(f"área de trocas, método em {b}", res)

    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(result, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"pronto: {out}/")


if __name__ == "__main__":
    main()

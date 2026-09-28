"""Leituras do espaco de representacao, pisos e agregacao -- comuns a tabular e a imagens.

As duas suites (`tabular_suite.py` e `pets_suite.py`) diferem so em como obtem as fontes
(MLP sobre vetores, ResNet sobre imagens); tudo o que se mede depois e este modulo. Cada
leitura e feita para a Gram conjunta, para cada fonte isolada e para o RSA euclidiano, e toda
leitura referida ao comportamento tem o piso cruzado ao lado (a mesma construcao na rede nao
treinada, com os mesmos pesos iniciais, contra o comportamento da rede treinada).

Leituras, por bloco:

  knn_pred             k-vizinhos leave-one-out contra a predicao
  correctness_auc      AUC da margem geometrica para o acerto
  confusion_soft       Spearman entre a proximidade dos pares de classes e a confusao suave
  confusion_test       o mesmo com a confusao contada no teste inteiro
  confusion            o mesmo com a confusao contada so na amostra (esparsa: ver abaixo)
  containment_k        fracao das direcoes de classe contidas em V_r, r = K − 1 para todas
  containment_matched  idem, r igual ao de A isolado
  containment          idem, r proprio de cada representacao (NAO comparavel entre elas)
  e_ratio_auc_k        AUC de Ê_ratio para o erro, r = K − 1
  cka_alignment        a = CKA(G, K_ativ)
  cka_seen_*/cka_resid_*  as metades da decomposicao Ĝ = a K̂ + R lidas como geometrias

Tres cuidados que vieram de erros medidos e corrigidos, e que o codigo impoe:

  dimensao   contencao e Ê_ratio dependem de r; a conjunta retem ~2,5 vezes mais modos que A
             nos blocos rasos, entao a comparacao entre representacoes so e feita com r fixo;
  confusao   a contagem na amostra deixa 80–96% dos pares de classes com zero em bases faceis;
             a leitura principal e a confusao suave, que usa a probabilidade e nao so os erros;
  empates    "conjunta > outra" e estrito, para que leituras saturadas nao contem como vitoria.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from .behaviour import class_distance_matrix, confusion_agreement, correctness_auc, cka, knn_loo
from .geometry import gram_distance, rsa_euclid
from .kernels import project_psd
from .sources import block_grams
from .spectral import _auc, _class_directions, eigendecompose, energy_readouts

REPS = ("joint", "activations", "grads", "params")
READINGS = (
    "knn_pred", "correctness_auc", "confusion", "confusion_test", "confusion_soft",
    "containment", "containment_k", "containment_matched", "e_ratio_auc", "e_ratio_auc_k",
    "cka_alignment", "cka_seen_knn", "cka_resid_knn", "cka_seen_auc", "cka_resid_auc",
    "cka_discarded",
)

# (rotulo, leitura, comparada com): "floor" = piso cruzado; um nome de representacao = a
# mesma leitura naquela representacao; "key:<leitura>" = outra leitura da propria conjunta.
CLAIMS = [
    ("kNN acima do piso", "knn_pred", "floor"),
    ("AUC do acerto acima do piso", "correctness_auc", "floor"),
    ("confusão suave acima do piso", "confusion_soft", "floor"),
    ("contenção (r = K−1) acima do piso", "containment_k", "floor"),
    ("Ê_ratio (r = K−1) acima do piso", "e_ratio_auc_k", "floor"),
    ("kNN: conjunta > só A", "knn_pred", "activations"),
    ("AUC do acerto: conjunta > só A", "correctness_auc", "activations"),
    ("confusão suave: conjunta > só A", "confusion_soft", "activations"),
    ("contenção (r = K−1): conjunta > só A", "containment_k", "activations"),
    ("contenção (r de A): conjunta > só A", "containment_matched", "activations"),
    ("Ê_ratio (r = K−1): conjunta > só A", "e_ratio_auc_k", "activations"),
    ("kNN: conjunta > RSA", "knn_pred", "rsa"),
    ("AUC do acerto: conjunta > RSA", "correctness_auc", "rsa"),
    ("CKA: metade residual prevê o acerto melhor que a vista", "cka_resid_auc", "key:cka_seen_auc"),
]

LABELS = {
    "knn_pred": "k-vizinhos contra a predição",
    "correctness_auc": "AUC da margem para o acerto",
    "confusion_soft": "Spearman com a confusão suave (probabilidade média, teste inteiro)",
    "confusion_test": "Spearman com a confusão contada no teste inteiro",
    "confusion": "Spearman com a confusão contada na amostra",
    "containment_k": "Contenção das classes em $V_r$, r = K − 1 para todas",
    "containment_matched": "Contenção das classes em $V_r$, r igual ao de A",
    "containment": "Contenção das classes em $V_r$, r próprio (não comparável entre representações)",
    "e_ratio_auc_k": "AUC de Ê_ratio para o erro, r = K − 1",
    "cka_alignment": "Alinhamento $a$ com a Gram das ativações (CKA)",
    "cka_seen_auc": "AUC do acerto na metade vista pela CKA",
    "cka_resid_auc": "AUC do acerto na metade residual",
    "cka_discarded": "Massa espectral descartada ao projetar a metade residual",
}


# --------------------------------------------------------------------------
# comportamento e intervencao
# --------------------------------------------------------------------------


@torch.no_grad()
def flip_rates(model, x: torch.Tensor, device: torch.device, sigmas, seed: int) -> dict:
    """Fracao de predicoes que mudam ao somar σ·std_c·N(0,1) a saida de cada bloco.

    O desvio e por canal: tomado sobre as amostras e, em mapas de convolucao, tambem sobre
    as posicoes -- entao σ e a mesma fracao do sinal em qualquer bloco e arquitetura.
    """
    model.eval()
    x = x.to(device)
    clean = model(x).argmax(1)
    store = {}
    handles = [model.get_block(b).register_forward_hook(lambda m, i, o, b=b: store.__setitem__(b, o))
               for b in model.block_names]
    model(x)
    for h in handles:
        h.remove()
    std = {}
    for b in model.block_names:
        o = store[b]
        dims = [0, *range(2, o.dim())]
        std[b] = o.std(dim=dims, keepdim=True)
    gen = torch.Generator(device=device).manual_seed(seed)
    out = {}
    for b in model.block_names:
        for sigma in sigmas:
            def noisy(_m, _i, o, b=b, sigma=sigma):
                return o + sigma * std[b] * torch.randn(o.shape, generator=gen, device=o.device)
            h = model.get_block(b).register_forward_hook(noisy)
            out[f"{b}@{sigma:g}"] = float((model(x).argmax(1) != clean).float().mean())
            h.remove()
    return out


def behaviour_references(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> dict:
    """Confusao contada e confusao suave, sobre todo o conjunto de avaliacao.

    A contagem sobre a amostra de 500 e esparsa demais em bases faceis (0,04 erro por par no
    letter). A confusao suave S_ab + S_ba, com S_ab a probabilidade media que o modelo da a b
    nas amostras de a, usa toda a amostra e nao so os erros.
    """
    preds = probs.argmax(1)
    soft = np.zeros((n_classes, n_classes))
    count = np.zeros((n_classes, n_classes))
    for a in range(n_classes):
        members = labels == a
        if members.any():
            soft[a] = probs[members].mean(0)
    for t, p in zip(labels, preds):
        count[t, p] += 1
    soft, count = soft + soft.T, count + count.T
    np.fill_diagonal(soft, 0.0)
    np.fill_diagonal(count, 0.0)
    iu = np.triu_indices(n_classes, k=1)
    return {"soft": soft, "test": count, "test_zero_pairs": float(np.mean(count[iu] == 0))}


def sample_zero_pairs(labels: np.ndarray, preds: np.ndarray, n_classes: int) -> float:
    counts = np.zeros((n_classes, n_classes))
    for t, p in zip(labels, preds):
        counts[t, p] += 1
    iu = np.triu_indices(n_classes, k=1)
    return float(np.mean((counts + counts.T)[iu] == 0))


def class_agreement(d: np.ndarray, labels: np.ndarray, n_classes: int, reference: np.ndarray) -> float:
    """Spearman entre a proximidade dos pares de classes e uma referencia de confusao."""
    m = class_distance_matrix(d, labels, n_classes)
    iu = np.triu_indices(n_classes, k=1)
    x, y = -m[iu], reference[iu]
    if not np.isfinite(x).all() or np.std(y) == 0 or np.std(x) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


# --------------------------------------------------------------------------
# leituras
# --------------------------------------------------------------------------


def fixed_rank_readings(gram, labels, preds, n_classes, ranks: dict, enough: bool) -> dict:
    """Contencao e Ê_ratio com o numero de modos fixado de fora (r = K − 1, r de A)."""
    vals, vecs = eigendecompose(gram)
    y = _class_directions(labels, n_classes)
    mode = vals[None, :] * vecs**2
    total = np.maximum(mode.sum(1), 1e-300)
    out = {}
    for name, r in ranks.items():
        if r is None:
            out[f"containment_{name}"] = float("nan")
            continue
        r = int(max(1, min(r, len(vals))))
        proj = vecs[:, :r].T @ y
        out[f"containment_{name}"] = float((proj**2).sum() / max((y**2).sum(), 1e-12))
        if name == "k":
            ratio = 1.0 - mode[:, :r].sum(1) / total
            out["e_ratio_auc_k"] = _auc(ratio, preds != labels) if enough else float("nan")
    return out


def _normalized(k: np.ndarray) -> np.ndarray:
    n = len(k)
    h = np.eye(n) - np.ones((n, n)) / n
    kc = h @ k @ h
    norm = np.linalg.norm(kc, "fro")
    return kc / (norm if norm > 0 else 1.0)


def cka_halves(gram: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Ĝ = a K̂ + R: a metade que a CKA ve, o residuo (ambos em PSD), a massa descartada e a.

    O residuo e uma diferenca de matrizes e nao e PSD; para ler distancias e preciso projeta-lo,
    e a fracao de massa espectral descartada nessa projecao vai junto -- sem ela a leitura da
    metade residual nao e auditavel (no CIFAR ela chegava a 48% no ultimo bloco).
    """
    g, k = _normalized(gram), _normalized(reference)
    a = float((g * k).sum())
    perp = g - a * k
    vals = np.linalg.eigvalsh(0.5 * (perp + perp.T))
    discarded = float(-vals[vals < 0].sum() / max(np.abs(vals).sum(), 1e-12))
    return project_psd(a * k), project_psd(perp), discarded, a


def readings(d, gram, labels, preds, margin, n_classes, cfg, act_gram=None, raw=None,
             r_match=None, refs=None, split=False) -> dict:
    """As leituras de uma representacao; NaN onde a condicao de validade falha.

    Uma representacao degenerada (Gram bruta com similaridade 1 entre todos os pares, como Γ
    no ultimo bloco sob alvo fixo) nao tem geometria e fica toda indefinida.
    """
    nan = float("nan")
    if raw is not None and float(np.min(raw)) > 1.0 - 1e-6:
        return {k: nan for k in (*READINGS, "e_ratio_cv", "rank")} | {"degenerate": True}
    errors = int((preds != labels).sum())
    enough = cfg.min_errors <= errors <= len(labels) - cfg.min_errors
    out = {
        "knn_pred": knn_loo(d, preds, cfg.k),
        "correctness_auc": correctness_auc(d, labels, preds, n_classes) if enough else nan,
        "confusion": confusion_agreement(d, labels, preds, n_classes)[0] if n_classes >= 5 else nan,
    }
    # com 3 ou 4 classes ha no maximo 6 pares: poucos demais para uma correlacao por postos
    for key, ref in (("confusion_test", "test"), ("confusion_soft", "soft")):
        out[key] = (class_agreement(d, labels, n_classes, refs[ref])
                    if refs is not None and n_classes >= 5 else nan)
    if gram is None:
        return out
    try:
        e = energy_readouts(gram, labels, preds, margin, n_classes, "participation", 0.95,
                            cfg.permutations, 0)
    except AssertionError:  # identidade G_ii = Σ λ V² quebrada: matriz sem geometria
        return out | {k: nan for k in READINGS if k not in out} | {"degenerate": True}
    out["containment"] = e["y_inside_Vr"]
    out["e_ratio_auc"] = e["e_ratio_error_auc"] if enough else nan
    out["e_ratio_cv"] = e["e_ratio_cv"]
    out["rank"] = e["rank_participation"]
    out["cka_alignment"] = cka(gram, act_gram) if act_gram is not None else nan
    out |= fixed_rank_readings(gram, labels, preds, n_classes,
                               {"k": n_classes - 1, "matched": r_match}, enough)
    if split and act_gram is not None:
        seen, resid, discarded, _ = cka_halves(gram, act_gram)
        for name, part in (("seen", seen), ("resid", resid)):
            dp = gram_distance(part)
            out[f"cka_{name}_knn"] = knn_loo(dp, preds, cfg.k)
            out[f"cka_{name}_auc"] = correctness_auc(dp, labels, preds, n_classes) if enough else nan
        out["cka_discarded"] = discarded
    return out


def analyse(sources: dict, sources_u: dict, blocks: list[str], labels, preds, logits, refs,
            stability: list[float], cfg, device) -> dict:
    """Todas as leituras de uma execucao, por bloco, com o piso cruzado."""
    n, n_classes = len(labels), logits.shape[1]
    top2 = np.sort(logits, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    metrics, floor = {}, {}
    for b in blocks:
        grams = block_grams(sources[b], cfg.shift, "trace_center", device)
        grams_u = block_grams(sources_u[b], cfg.shift, "trace_center", device)
        a = sources[b]["activations"].reshape(n, -1).astype(np.float64)
        a_u = sources_u[b]["activations"].reshape(n, -1).astype(np.float64)
        act_gram, act_gram_u = a @ a.T, a_u @ a_u.T
        # A primeiro: o seu r e o numero de modos da comparacao "com o mesmo r"
        row = {"activations": readings(gram_distance(grams["activations"]), grams["activations"],
                                       labels, preds, margin, n_classes, cfg, act_gram,
                                       grams["_raw"]["activations"], refs=refs)}
        r_a = row["activations"].get("rank")
        row["activations"] |= fixed_rank_readings(grams["activations"], labels, preds, n_classes,
                                                  {"matched": r_a}, True)
        for rep in ("joint", "grads", "params"):
            row[rep] = readings(gram_distance(grams[rep]), grams[rep], labels, preds, margin,
                                n_classes, cfg, act_gram, grams["_raw"][rep], r_a, refs,
                                split=(rep == "joint"))
        row["rsa"] = readings(rsa_euclid(sources[b]["activations"]), None, labels, preds, margin,
                              n_classes, cfg, refs=refs)
        metrics[b] = row
        floor[b] = readings(gram_distance(grams_u["joint"]), grams_u["joint"], labels, preds,
                            margin, n_classes, cfg, act_gram_u, grams_u["_raw"]["joint"], r_a,
                            refs, split=True)
    curve = [metrics[b]["joint"]["knn_pred"] for b in blocks]
    return {"metrics": metrics, "floor": floor,
            "dating_spearman": float(spearmanr(stability, curve).statistic)}


# --------------------------------------------------------------------------
# execucao, agregacao e relatorio
# --------------------------------------------------------------------------


def run_all(keys: list[tuple[str, int]], run_fn, out: Path) -> list[dict]:
    """Executa cada (unidade, semente), com retomada e gravacao atomica."""
    (out / "runs").mkdir(parents=True, exist_ok=True)
    runs = []
    for name, seed in keys:
        path = out / "runs" / f"{name}_s{seed}.json"
        if path.exists():
            runs.append(json.loads(path.read_text()))
            continue
        t = time.time()
        r = run_fn(name, seed)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, ensure_ascii=False))
        tmp.replace(path)  # uma interrupcao no meio nao deixa um arquivo truncado
        runs.append(r)
        v = r["variants"]
        print(f"{name:14s} semente {seed}: teste {r['test_acc']:.3f}, erros na amostra "
              f"{r['sample_errors']:3d}, datação ρ predito {v['pred']['dating_spearman']:+.2f} "
              f"fixo {v['fixed']['dating_spearman']:+.2f} ({time.time() - t:.0f}s)", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return runs


def view(runs: list[dict], variant: str) -> list[dict]:
    return [{**r, **r["variants"][variant]} for r in runs]


def invariants(runs: list[dict]) -> dict:
    """Fracao das execucoes (por bloco) em que cada afirmacao de `CLAIMS` se mantem."""
    blocks = runs[0]["blocks"]
    out = {}
    for label, reading, against in CLAIMS:
        out[label] = {}
        for b in blocks:
            values = []
            for r in runs:
                m = r["metrics"][b]
                mine = m["joint"].get(reading, np.nan)
                if against == "floor":
                    other = r["floor"][b].get(reading, np.nan)
                elif against.startswith("key:"):
                    other = m["joint"].get(against[4:], np.nan)
                else:
                    other = m[against].get(reading, np.nan)
                if np.isfinite(mine) and np.isfinite(other):
                    values.append(mine > other)
            out[label][b] = (float(np.mean(values)) if values else float("nan"), len(values))
    return out


def summarize(runs: list[dict], order: list[str]) -> dict:
    """Media e desvio por unidade (entre sementes) e entre unidades (das medias)."""
    blocks = runs[0]["blocks"]
    names = sorted({r["dataset"] for r in runs}, key=order.index)
    per = {}
    for n in names:
        rs = [r for r in runs if r["dataset"] == n]
        entry = {"test_acc": (np.mean([r["test_acc"] for r in rs]), np.std([r["test_acc"] for r in rs])),
                 "sample_errors": float(np.mean([r["sample_errors"] for r in rs])),
                 "dating_spearman": (np.nanmean([r["dating_spearman"] for r in rs]),
                                     np.nanstd([r["dating_spearman"] for r in rs])),
                 "n_classes": rs[0]["n_classes"], "n_runs": len(rs),
                 "zero_pairs_sample": float(np.mean([r.get("sample_zero_pairs", np.nan) for r in rs])),
                 "zero_pairs_test": float(np.mean([r.get("test_zero_pairs", np.nan) for r in rs]))}
        for reading in READINGS:
            for source in ("joint", "activations", "grads", "rsa", "floor"):
                vals = np.array([[(r["floor"][b] if source == "floor" else r["metrics"][b][source]).get(reading, np.nan)
                                  for b in blocks] for r in rs], dtype=float)
                with np.errstate(all="ignore"):
                    entry[f"{reading}/{source}"] = (np.nanmean(vals, 0).tolist(), np.nanstd(vals, 0).tolist())
        per[n] = entry
    across = {}
    for reading in READINGS:
        for source in ("joint", "activations", "grads", "rsa", "floor"):
            means = np.array([per[n][f"{reading}/{source}"][0] for n in names], dtype=float)
            with np.errstate(all="ignore"):
                across[f"{reading}/{source}"] = (np.nanmean(means, 0).tolist(), np.nanstd(means, 0).tolist(),
                                                 np.sum(~np.isnan(means), 0).tolist())
    return {"blocks": blocks, "datasets": names, "per_dataset": per, "across": across,
            "invariants": invariants(runs)}


def _pm(m, s, digits=3):
    if m is None or not np.isfinite(m):
        return "—"
    return f"{m:.{digits}f} ± {s:.{digits}f}".replace(".", ",")


def write_tables(summary: dict, path: Path, title: str) -> None:
    blocks, names = summary["blocks"], summary["datasets"]
    lines = [f"# {title}\n",
             ("Média ± desvio padrão entre sementes, por base; ao final de cada tabela, média ± desvio "
              "**entre bases** das médias por base, com A, Γ, RSA e o piso cruzado.\n"
              if len(summary["datasets"]) > 1 else
              "Média ± desvio padrão **entre sementes**; com uma base só não há desvio entre bases, "
              "então as linhas de A, Γ, RSA e do piso cruzado também são entre sementes.\n"),
             "## Bases\n",
             "| base | classes | acurácia de teste | erros na amostra | pares sem confusão (amostra) "
             "| pares sem confusão (teste) | Spearman da datação |",
             "|---|---|---|---|---|---|---|"]
    for n in names:
        e = summary["per_dataset"][n]
        zs, zt = e.get("zero_pairs_sample", np.nan), e.get("zero_pairs_test", np.nan)
        lines.append(f"| {n} | {e['n_classes']} | {_pm(*e['test_acc'])} | {e['sample_errors']:.0f} | "
                     f"{zs:.0%} | {zt:.0%} | {_pm(*e['dating_spearman'], 2)} |".replace("nan%", "—"))
    lines.append("")
    for reading, label in LABELS.items():
        lines += [f"## {label}\n", "| base | " + " | ".join(blocks) + " |", "|---|" + "---|" * len(blocks)]
        for n in names:
            m, s = summary["per_dataset"][n][f"{reading}/joint"]
            lines.append(f"| {n} | " + " | ".join(_pm(a, b) for a, b in zip(m, s)) + " |")
        single = len(names) == 1
        for source, name in (("joint", "**média entre bases (conjunta)**"), ("activations", "média (só A)"),
                             ("grads", "média (só Γ)"), ("rsa", "média (RSA)"),
                             ("floor", "média (piso cruzado)")):
            if single:
                if source == "joint":
                    continue  # seria a repeticao exata da linha da base, com desvio zero
                m, s = summary["per_dataset"][names[0]][f"{reading}/{source}"]
            else:
                m, s, _ = summary["across"][f"{reading}/{source}"]
            if all(not np.isfinite(v) for v in m):
                continue
            lines.append(f"| {name} | " + " | ".join(_pm(a, b) for a, b in zip(m, s)) + " |")
        lines.append("")
    lines += ["## Em que fração das execuções cada afirmação se mantém\n",
              "Fração (e número de execuções válidas) por bloco.\n",
              "| afirmação | " + " | ".join(blocks) + " |", "|---|" + "---|" * len(blocks)]
    for claim, per_block in summary["invariants"].items():
        cells = []
        for b in blocks:
            frac, count = per_block[b]
            cells.append("—" if not np.isfinite(frac) else f"{frac:.2f} ({count})".replace(".", ","))
        lines.append(f"| {claim} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def write_comparison(summaries: dict, path: Path) -> None:
    """Alvo predito × alvo fixo: o que sobrevive ao controle da tautologia de Γ."""
    blocks = summaries["pred"]["blocks"]
    lines = ["# Alvo predito × alvo fixo\n",
             "Com o alvo predito, Γ carrega a predição (parcialmente nos blocos rasos, totalmente no "
             "último); com o alvo fixo, esse condicionamento sai. Média ± desvio entre bases.\n"]
    for reading in ("knn_pred", "correctness_auc", "containment_k", "confusion_soft"):
        lines += [f"## {LABELS[reading]}\n", "| representação | alvo | " + " | ".join(blocks) + " |",
                  "|---|---|" + "---|" * len(blocks)]
        for source, label in (("joint", "conjunta"), ("grads", "só Γ"), ("activations", "só A"),
                              ("floor", "piso cruzado")):
            for variant in ("pred", "fixed"):
                if source == "activations" and variant == "fixed":
                    continue  # A nao depende do alvo da margem
                m, s, _ = summaries[variant]["across"][f"{reading}/{source}"]
                lines.append(f"| {label} | {'predito' if variant == 'pred' else 'fixo'} | "
                             + " | ".join(_pm(a, b) for a, b in zip(m, s)) + " |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def report(runs: list[dict], out: Path, order: list[str], title: str, plot_fn) -> dict:
    summaries = {}
    for variant in ("pred", "fixed"):
        s = summarize(view(runs, variant), order)
        summaries[variant] = s
        (out / f"summary_{variant}.json").write_text(json.dumps(s, ensure_ascii=False, default=float))
        write_tables(s, out / f"tables_{variant}.md", f"{title} — alvo {'predito' if variant == 'pred' else 'fixo'}")
        plot_fn(s, str(out / f"suite_{variant}.png"))
    write_comparison(summaries, out / "comparison.md")
    return summaries

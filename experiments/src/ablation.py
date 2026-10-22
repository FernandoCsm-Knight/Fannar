"""Ablacoes da composicao e referencias de confianca (Subsecao `subsec:adicionais`, itens a-c).

A suite (`suite.py`) compara a Gram conjunta com cada fonte isolada, o que mostra se *juntar*
as fontes ajuda, mas nao se a forma *multiplicativa* e o que ajuda, nem se a normalizacao
angular T_a e o deslocamento por J importam. Este modulo constroi, com as mesmas fontes, as
alternativas que isolam cada escolha:

  joint       G = Π_l (T_a(K_l) + J)/2            a composicao do artigo
  additive    G = (1/d) Σ_l (T_a(K_l) + J)/2      a soma, com a mesma T_a e o mesmo J
  no_angular  G = Π_l (K_l/m_l + J)/2             sem T_a (m_l = media da diagonal de K_l)
  no_shift    G = Π_l T_a(K_l)                    sem o deslocamento por J
  no_both     G = Π_l K_l/m_l                     sem T_a e sem J

Soma e concatenacao. Com kernel linear e fontes normalizadas, a Gram angular da concatenacao
[u_1/|u_1|, ..., u_d/|u_d|] e exatamente a media das Grams angulares das fontes (o produto
interno da concatenacao e a soma dos cossenos, e a norma e √d). Depois do deslocamento, a
"Gram da concatenacao" e a "soma ponderada com pesos iguais" sao a mesma matriz, e `additive`
cobre as duas.

Todas passam pela mesma transformacao final da conjunta (T_tr ∘ T_c) e pelas mesmas leituras
de `suite.readings`, e cada uma tem o seu piso cruzado (a mesma construcao na rede nao
treinada). Alem disso:

  item (b)  a AUC da margem geometrica para o acerto e comparada a sinais de confianca que nao
            precisam do metodo: probabilidade maxima do softmax (Hendrycks & Gimpel, 2017),
            margem entre os dois maiores logits, trust score (Jiang et al., 2018, sem a
            filtragem por densidade, alpha = 0) e distancia de Mahalanobis com covariancia
            compartilhada (Lee et al., 2018), as duas ultimas nas ativacoes do ultimo bloco
            com medias de classe tiradas do treino;
  item (c)  k-vizinhos contra a predicao na distancia euclidiana dos logits e das ativacoes do
            ultimo bloco, que com o colapso neural recuperam a predicao quase por construcao.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

from . import suite as S
from .geometry import gram_distance, inner_product, rsa_euclid
from .kernels import apply_transform, hadamard, project_psd
from .sources import SOURCE_KEYS
from .spectral import _auc

COMPOSITIONS = ("joint", "additive", "no_angular", "no_shift", "no_both")
SINGLES = ("activations", "grads")
REPS = (*COMPOSITIONS, *SINGLES)
READINGS = ("knn_pred", "correctness_auc", "confusion_soft", "containment_k", "e_ratio_auc_k")
CONFIDENCE = ("softmax", "logit_margin", "trust_score", "mahalanobis")

LABEL = {
    "joint": r"produto (conjunta)",
    "additive": r"soma (= concatenação)",
    "no_angular": r"produto sem $T_a$",
    "no_shift": r"produto sem $\mathbf{J}$",
    "no_both": r"produto sem $T_a$ e sem $\mathbf{J}$",
    "activations": r"só $A$",
    "grads": r"só $\Omega$",
    "softmax": "softmax máximo",
    "logit_margin": "margem dos logits",
    "trust_score": "trust score",
    "mahalanobis": "Mahalanobis",
    "geometric": "margem geométrica",
    "knn_logits": "k-vizinhos nos logits",
    "knn_last": "k-vizinhos nas ativações finais",
}


# --------------------------------------------------------------------------
# composicoes
# --------------------------------------------------------------------------


def _shifted(k: np.ndarray, shift: float) -> np.ndarray:
    return (k + shift) / (1.0 + shift)


def composition_grams(block_sources: dict, shift: float = 1.0, device=None) -> dict:
    """As cinco composicoes e as duas fontes isoladas de um bloco, ja transformadas.

    Como em `sources.block_grams`, cada matriz passa por `project_psd` e por T_tr ∘ T_c; as
    matrizes brutas com diagonal 1 vao em `_raw`, que e o que `suite.readings` usa para
    reconhecer uma representacao degenerada.
    """
    inner = {k: inner_product(block_sources[k], device) for k in SOURCE_KEYS}
    angular = {k: apply_transform(v, "angular") for k, v in inner.items()}
    linear = {k: v / max(float(np.mean(np.diag(v))), 1e-12) for k, v in inner.items()}

    raw = {
        "joint": hadamard([_shifted(angular[k], shift) for k in SOURCE_KEYS]),
        "additive": np.mean([_shifted(angular[k], shift) for k in SOURCE_KEYS], axis=0),
        "no_angular": hadamard([_shifted(linear[k], shift) for k in SOURCE_KEYS]),
        "no_shift": hadamard([angular[k] for k in SOURCE_KEYS]),
        "no_both": hadamard([linear[k] for k in SOURCE_KEYS]),
        "activations": _shifted(angular["activations"], shift),
        "grads": _shifted(angular["grads"], shift),
    }
    out = {k: apply_transform(project_psd(v), "trace_center") for k, v in raw.items()}
    # sem T_a a diagonal nao e 1 e o teste de degenerescencia de `readings` nao se aplica
    out["_raw"] = {k: (v if k not in ("no_angular", "no_both") else None) for k, v in raw.items()}
    return out


# --------------------------------------------------------------------------
# confianca
# --------------------------------------------------------------------------


def _sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d2 = (a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2.0 * a @ b.T
    return np.maximum(d2, 0.0)


def trust_score(feats: np.ndarray, preds: np.ndarray, train: np.ndarray, train_labels: np.ndarray,
                n_classes: int) -> np.ndarray:
    """d(x, classe vizinha mais proxima) / d(x, classe predita), no treino (Jiang et al., 2018)."""
    nearest = np.full((len(feats), n_classes), np.inf)
    for c in range(n_classes):
        members = train[train_labels == c]
        if len(members):
            nearest[:, c] = np.sqrt(_sq_dist(feats, members).min(1))
    rows = np.arange(len(feats))
    own = nearest[rows, preds]
    other = nearest.copy()
    other[rows, preds] = np.inf
    return other.min(1) / np.maximum(own, 1e-12)


def mahalanobis_score(feats: np.ndarray, train: np.ndarray, train_labels: np.ndarray,
                      n_classes: int) -> np.ndarray:
    """−min_c (x − μ_c)ᵀ Σ⁻¹ (x − μ_c), Σ compartilhada e encolhida (Ledoit-Wolf)."""
    present = [c for c in range(n_classes) if (train_labels == c).any()]
    means = np.stack([train[train_labels == c].mean(0) for c in present])
    centred = train - means[np.searchsorted(present, train_labels)]
    precision = LedoitWolf().fit(centred).precision_
    dist = np.stack([np.einsum("nd,de,ne->n", feats - m, precision, feats - m) for m in means], 1)
    return -dist.min(1)


def confidence_aucs(logits: np.ndarray, labels: np.ndarray, feats: np.ndarray, train: np.ndarray,
                    train_labels: np.ndarray) -> dict:
    """AUC de cada sinal de confianca para o acerto, na mesma amostra das leituras."""
    n_classes = logits.shape[1]
    preds = logits.argmax(1)
    correct = preds == labels
    z = np.exp(logits - logits.max(1, keepdims=True))
    top2 = np.sort(logits, axis=1)[:, -2:]
    scores = {
        "softmax": (z / z.sum(1, keepdims=True)).max(1),
        "logit_margin": top2[:, 1] - top2[:, 0],
        "trust_score": trust_score(feats, preds, train, train_labels, n_classes),
        "mahalanobis": mahalanobis_score(feats, train, train_labels, n_classes),
    }
    return {k: _auc(v, correct) for k, v in scores.items()}


# --------------------------------------------------------------------------
# uma execucao
# --------------------------------------------------------------------------


def _subset(r: dict) -> dict:
    return {k: r.get(k, float("nan")) for k in (*READINGS, "degenerate")}


def analyse(sources: dict, sources_u: dict, blocks: list[str], labels, preds, logits, refs,
            feats: np.ndarray, train_feats: np.ndarray, train_labels: np.ndarray, cfg, device) -> dict:
    """Leituras de cada composicao por bloco, com piso cruzado, e as referencias de (b) e (c)."""
    n_classes = logits.shape[1]
    top2 = np.sort(logits, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    metrics, floor = {}, {}
    for b in blocks:
        g = composition_grams(sources[b], cfg.shift, device)
        g_u = composition_grams(sources_u[b], cfg.shift, device)
        metrics[b] = {rep: _subset(S.readings(gram_distance(g[rep]), g[rep], labels, preds, margin,
                                              n_classes, cfg, raw=g["_raw"][rep], refs=refs))
                      for rep in REPS}
        floor[b] = {rep: _subset(S.readings(gram_distance(g_u[rep]), g_u[rep], labels, preds, margin,
                                            n_classes, cfg, raw=g_u["_raw"][rep], refs=refs))
                    for rep in COMPOSITIONS}
    last = sources[blocks[-1]]["activations"]
    baselines = {
        "knn_logits": _subset(S.readings(rsa_euclid(logits[:, :, None]), None, labels, preds, margin,
                                         n_classes, cfg, refs=refs)),
        "knn_last": _subset(S.readings(rsa_euclid(last), None, labels, preds, margin, n_classes, cfg,
                                       refs=refs)),
    }
    return {"metrics": metrics, "floor": floor, "baselines": baselines,
            "confidence": confidence_aucs(logits, labels, feats, train_feats, train_labels)}


# --------------------------------------------------------------------------
# agregacao
# --------------------------------------------------------------------------


def summarize(runs: list[dict]) -> dict:
    """Media entre bases (das medias entre sementes) e fracao das execucoes em que a conjunta
    supera estritamente cada alternativa e o proprio piso."""
    blocks = runs[0]["blocks"]
    names = list(dict.fromkeys(r["dataset"] for r in runs))

    def per_dataset(get):
        vals = np.array([[np.nanmean([get(r, b) for r in runs if r["dataset"] == n]) for b in blocks]
                         for n in names], dtype=float)
        with np.errstate(all="ignore"):
            return np.nanmean(vals, 0).tolist(), np.nanstd(vals, 0).tolist()

    out = {"blocks": blocks, "datasets": names, "n_runs": len(runs), "mean": {}, "floor": {}, "wins": {},
           "above_floor": {}}
    for reading in READINGS:
        for rep in REPS:
            out["mean"][f"{reading}/{rep}"] = per_dataset(lambda r, b: r["metrics"][b][rep][reading])
        for rep in COMPOSITIONS:
            out["floor"][f"{reading}/{rep}"] = per_dataset(lambda r, b: r["floor"][b][rep][reading])
        for rep in REPS[1:]:
            wins = []
            for b in blocks:
                pairs = [(r["metrics"][b]["joint"][reading], r["metrics"][b][rep][reading]) for r in runs]
                pairs = [(x, y) for x, y in pairs if np.isfinite(x) and np.isfinite(y)]
                wins.append(float(np.mean([x > y for x, y in pairs])) if pairs else float("nan"))
            out["wins"][f"{reading}/{rep}"] = wins
        for rep in COMPOSITIONS:
            above = []
            for b in blocks:
                pairs = [(r["metrics"][b][rep][reading], r["floor"][b][rep][reading]) for r in runs]
                pairs = [(x, y) for x, y in pairs if np.isfinite(x) and np.isfinite(y)]
                above.append(float(np.mean([x > y for x, y in pairs])) if pairs else float("nan"))
            out["above_floor"][f"{reading}/{rep}"] = above
    out["baselines"] = {k: {reading: float(np.nanmean([r["baselines"][k][reading] for r in runs]))
                            for reading in READINGS[:3]} for k in ("knn_logits", "knn_last")}
    out["confidence"] = {k: (float(np.nanmean([r["confidence"][k] for r in runs])),
                             float(np.nanstd([r["confidence"][k] for r in runs]))) for k in CONFIDENCE}
    out["geometric"] = {b: (float(np.nanmean([r["metrics"][b]["joint"]["correctness_auc"] for r in runs])),
                            float(np.nanstd([r["metrics"][b]["joint"]["correctness_auc"] for r in runs])))
                        for b in blocks}
    return out


def _f(v, d=3) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.{d}f}".replace(".", ",")


def write_tables(summary: dict, path, title: str) -> None:
    blocks = summary["blocks"]
    head = "| | " + " | ".join(blocks) + " |"
    rule = "|---|" + "---|" * len(blocks)
    L = [f"# {title}\n",
         f"{summary['n_runs']} execuções ({len(summary['datasets'])} bases), alvo fixo. Média entre bases das "
         "médias entre sementes; com uma base só, entre sementes.\n"]
    for reading in READINGS:
        L += [f"## {S.LABELS.get(reading, reading)}\n", head, rule]
        for rep in REPS:
            L.append(f"| {LABEL[rep]} | " + " | ".join(_f(v) for v in summary["mean"][f"{reading}/{rep}"][0]) + " |")
        for rep in COMPOSITIONS:
            L.append(f"| piso: {LABEL[rep]} | "
                     + " | ".join(_f(v) for v in summary["floor"][f"{reading}/{rep}"][0]) + " |")
        L += ["", "Fração das execuções em que a conjunta supera estritamente cada alternativa:\n", head, rule]
        for rep in REPS[1:]:
            L.append(f"| > {LABEL[rep]} | " + " | ".join(_f(v, 2) for v in summary["wins"][f"{reading}/{rep}"]) + " |")
        L += ["", "Fração das execuções acima do próprio piso:\n", head, rule]
        for rep in COMPOSITIONS:
            L.append(f"| {LABEL[rep]} | "
                     + " | ".join(_f(v, 2) for v in summary["above_floor"][f"{reading}/{rep}"]) + " |")
        L.append("")
    L += ["## Item (b): previsão do acerto (AUC)\n", "| sinal | AUC |", "|---|---|"]
    for b in blocks:
        m, s = summary["geometric"][b]
        L.append(f"| margem geométrica, conjunta em {b} | {_f(m)} ± {_f(s)} |")
    for k in CONFIDENCE:
        m, s = summary["confidence"][k]
        L.append(f"| {LABEL[k]} | {_f(m)} ± {_f(s)} |")
    L += ["", "## Item (c): vizinhança sem o método\n", "| referência | k-vizinhos | AUC do acerto |", "|---|---|---|"]
    for k in ("knn_logits", "knn_last"):
        v = summary["baselines"][k]
        L.append(f"| {LABEL[k]} | {_f(v['knn_pred'])} | {_f(v['correctness_auc'])} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")

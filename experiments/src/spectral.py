"""Energia espectral por amostra (theory.tex, subsec:spectral).

Como a energia e aferida
------------------------
De $\\mathbf{G}=V\\Lambda V^\\top$ segue, por decomposicao espectral e para *qualquer*
matriz simetrica,

    Ê_ik = λ̂_k V_ik²,        G_ii = Σ_k Ê_ik,
    Ê_i∥ = Σ_{k≤r} Ê_ik,     Ê_i⊥ = Σ_{k>r} Ê_ik,
    Ê_ratio(z_i) = Ê_i⊥ / G_ii,

que sao as eqs. `empirical_mode_energy`, `empirical_energies` e `empirical_e_ratio`. A
aferição e exata; nao ha aproximação nem parametro livre alem de r. `energies` verifica
a identidade `G_ii = Σ_k Ê_ik` a cada chamada, justamente porque ela e o unico ponto em
que um erro de sinal ou de ordenacao passaria silencioso.

O que a escolha de T muda -- e o que precisa ser reportado
---------------------------------------------------------
A transformacao nao afeta a aritmetica acima, e afeta *de qual vetor* a energia e a
norma:

  Gram bruta (angular + deslocamento, diagonal 1)
      E_total(z_i) = ‖Φ(z_i)‖²_{H_K} = 1. E a energia do RKHS no sentido estrito da
      teoria, mas constante por construcao: a normalizacao angular fixa a norma de
      toda amostra. E_par e E_perp perdem a leitura de magnitude ("quais amostras tem
      mais energia"); E_ratio, sendo uma **razao**, sobrevive, porque mede a direcao
      de Φ(z_i) relativa a V_r e nao o seu tamanho.

  Gram transformada (T_tr ∘ T_c)
      G_ii = ‖φ_i - φ̄‖² / tr, com φ_i os vetores do espaco de caracteristicas
      *empirico* que a `eq:empirical_feature_realization` garante existir para toda
      G ∈ S_+^C. Continua sendo energia -- norma ao quadrado numa realizacao de
      produto interno -- mas nao a de Φ(z_i) em H_K.

Por isso `energy_readouts` mede nas duas e o relatorio traz as duas. Afirmar «energia
do RKHS» sobre a Gram transformada, sem essa ressalva, seria trocar um objeto pelo
outro.

Selecao de r
------------
Padrao: razao de participacao, $(\\sum\\lambda)^2/\\sum\\lambda^2$, que nao tem limiar a
escolher e conta quantos modos estao de fato ativos. O truncamento por energia
acumulada em tau (`eq:r_threshold`) e reportado ao lado, nunca em vez dele.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def eigendecompose(gram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Autovalores em ordem decrescente, clipados em zero, e seus autovetores."""
    vals, vecs = np.linalg.eigh(0.5 * (gram + gram.T))
    order = np.argsort(vals)[::-1]
    return np.maximum(vals[order], 0.0), vecs[:, order]


def select_rank(vals: np.ndarray, rule: str = "participation", tau: float = 0.95) -> int:
    """Numero de modos retidos. `participation` nao tem limiar; `tau` e a eq. r_threshold."""
    total = float(vals.sum())
    if total <= 0:
        return 1
    if rule == "tau":
        return int(np.searchsorted(np.cumsum(vals) / total, tau) + 1)
    return max(1, int(round(total**2 / float((vals**2).sum()))))


def energies(gram: np.ndarray, rule: str = "participation", tau: float = 0.95) -> dict:
    """E_ik, E_total, E_par, E_perp e E_ratio por amostra."""
    vals, vecs = eigendecompose(gram)
    mode = vals[None, :] * vecs**2  # (n, n): Ê_ik
    r = select_rank(vals, rule, tau)
    total = mode.sum(1)
    # a identidade G_ii = Σ_k Ê_ik e o teste de sanidade da aferição
    if not np.allclose(total, np.diag(gram), atol=1e-8, rtol=1e-6):
        raise AssertionError("G_ii != sum_k lambda_k V_ik^2: decomposição inconsistente")
    par = mode[:, :r].sum(1)
    perp = mode[:, r:].sum(1)
    safe = np.maximum(total, 1e-300)
    return {
        "eigenvalues": vals,
        "eigenvectors": vecs,
        "mode_energy": mode,
        "rank": r,
        "rank_tau": select_rank(vals, "tau", tau),
        "E_total": total,
        "E_par": par,
        "E_perp": perp,
        "E_ratio": perp / safe,
        "retained_fraction": float(vals[:r].sum() / max(vals.sum(), 1e-300)),
    }


# --------------------------------------------------------------------------
# onde a classe mora no espectro
# --------------------------------------------------------------------------


def _class_directions(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Indicadores de classe centrados e normalizados, uma coluna por classe."""
    y = np.zeros((len(labels), n_classes))
    y[np.arange(len(labels)), labels] = 1.0
    y = y - y.mean(0, keepdims=True)
    norms = np.linalg.norm(y, axis=0, keepdims=True)
    return y / np.maximum(norms, 1e-12)


def class_energy(
    gram: np.ndarray,
    labels: np.ndarray,
    n_classes: int = 10,
    permutations: int = 50,
    seed: int = 0,
    vals: np.ndarray | None = None,
    vecs: np.ndarray | None = None,
) -> dict:
    """rho_classe = Σ_c y_c^T G y_c / tr(G), e a energia de classe modo a modo.

    E a fracao da energia espectral que a camada gasta nas direcoes de classe. Como
    y vem do **rotulo verdadeiro**, rho nao passa pela decisao da rede; o acaso vem
    de permutar os rotulos. E a versao em traco daquilo que o CKA mede em Frobenius,
    e as duas nao coincidem: o CKA normaliza por ‖G‖_F = sqrt(Σλ²), rho por
    tr(G) = Σλ.
    """
    if vals is None or vecs is None:
        vals, vecs = eigendecompose(gram)
    trace = float(np.trace(gram)) or 1.0
    y = _class_directions(labels, n_classes)
    rho = float(np.einsum("ic,ij,jc->", y, gram, y) / trace)
    projections = vecs.T @ y  # (modo, classe)
    per_mode = vals * (projections**2).sum(1)

    rng = np.random.default_rng(seed)
    floor = []
    for _ in range(permutations):
        yp = _class_directions(rng.permutation(labels), n_classes)
        floor.append(float(np.einsum("ic,ij,jc->", yp, gram, yp) / trace))
    mean_floor = float(np.mean(floor)) if floor else float("nan")
    return {
        "rho_class": rho,
        "rho_class_floor": mean_floor,
        "rho_class_ratio": rho / max(mean_floor, 1e-12),
        "class_energy_per_mode": per_mode,
        "class_energy_in_mode_1": float(per_mode[0] / max(per_mode.sum(), 1e-300)),
        "y_inside_Vr": None,  # preenchido por `energy_readouts`, que conhece r
    }


# --------------------------------------------------------------------------
# leituras comportamentais em linguagem de energia
# --------------------------------------------------------------------------


def _auc(score: np.ndarray, positive: np.ndarray) -> float:
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def energy_readouts(
    gram: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    logit_margin: np.ndarray,
    n_classes: int = 10,
    rule: str = "participation",
    tau: float = 0.95,
    permutations: int = 50,
    seed: int = 0,
) -> dict:
    """Tudo o que se le do espectro de uma Gram, incluindo o vinculo com a rede.

    As leituras comportamentais aqui sao **independentes das distancias**: as outras
    do projeto leem d_G, estas leem o espectro. E o vinculo que faltava entre a
    `subsec:spectral` e o comportamento do modelo.
    """
    e = energies(gram, rule, tau)
    ce = class_energy(
        gram, labels, n_classes, permutations, seed, e["eigenvalues"], e["eigenvectors"]
    )
    ratio = e["E_ratio"]
    r = e["rank"]
    y = _class_directions(labels, n_classes)
    inside = e["eigenvectors"][:, :r].T @ y  # projecao das direcoes de classe em V_r
    ce["y_inside_Vr"] = float((inside**2).sum() / max((y**2).sum(), 1e-12))
    return {
        "rank_participation": r,
        "rank_tau": e["rank_tau"],
        "retained_fraction": e["retained_fraction"],
        "e_ratio_mean": float(ratio.mean()),
        "e_ratio_cv": float(ratio.std() / max(abs(ratio.mean()), 1e-12)),
        "e_par_cv": float(e["E_par"].std() / max(abs(e["E_par"].mean()), 1e-12)),
        "e_ratio_margin_spearman": float(spearmanr(ratio, logit_margin).statistic),
        "e_ratio_error_auc": _auc(ratio, preds != labels),
        "rho_class": ce["rho_class"],
        "rho_class_floor": ce["rho_class_floor"],
        "rho_class_ratio": ce["rho_class_ratio"],
        "class_energy_in_mode_1": ce["class_energy_in_mode_1"],
        "y_inside_Vr": ce["y_inside_Vr"],
        "_spectrum": e["eigenvalues"],
        "_class_energy_per_mode": ce["class_energy_per_mode"],
        "_e_ratio": ratio,
    }

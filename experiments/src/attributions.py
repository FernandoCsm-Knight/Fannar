"""Atribuicoes contrastivas de referencia para a leitura de pares (P6).

  integrated_gradients  gradientes integrados (Sundararajan et al., 2017) da margem
                        m = f_{c_j} − f_{c_i} no caminho reto de x_i a x_j. Pela completude,
                        Σ_f φ_f = m(x_j) − m(x_i): e a mesma quantidade que o Shapley com j de
                        referencia decompoe, por outro caminho (gradiente em vez de coalizoes).
  dice                  DiCE (Mothilal et al., 2020), metodo `random`, que so precisa de
                        `predict_proba`: gera contrafactuais de x_i para a classe de j, e a
                        pontuacao de cada atributo e a fracao dos contrafactuais que o alteram.
                        Os empates (frequencias iguais) sao desfeitos ao acaso, nunca pelo indice.
                        `dice_counterfactuals` devolve os proprios contrafactuais gerados.
  baseline_shapley      Shapley exato do jogo v(S) = m(x_j em S, x_i fora) − m(x_i), m = f_{c_j} − f_{c_i}
                        (baseline Shapley, Sundararajan & Najmi 2020), por enumeracao dos 2^F
                        subconjuntos na GPU; `sampled_shapley` o estima por permutacoes.
  library_shap          SHAP intervencional da biblioteca (PermutationExplainer).
  lime_contrastive      LIME em regressao sobre a margem m, amostrando em torno de x_i;
                        `lime_weights` e o LIME de classificacao usual.
"""

from __future__ import annotations

from math import factorial
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from lime.lime_tabular import LimeTabularExplainer


def integrated_gradients(model, xi, xj, ci: int, cj: int, device, steps: int = 64) -> np.ndarray:
    """φ_f = (x_{j,f} − x_{i,f}) ∫ ∂m/∂x_f (x_i + α(x_j − x_i)) dα, regra do ponto medio."""
    alphas = (torch.arange(steps, device=device, dtype=torch.float32) + 0.5) / steps
    a = torch.as_tensor(np.asarray(xi), dtype=torch.float32, device=device)
    b = torch.as_tensor(np.asarray(xj), dtype=torch.float32, device=device)
    path = (a + alphas[:, None] * (b - a)).requires_grad_(True)
    model.eval()
    logits = model(path)
    (logits[:, cj] - logits[:, ci]).sum().backward()
    return (path.grad.mean(0) * (b - a)).detach().cpu().numpy().astype(np.float64)


class _Wrapped:
    """O modelo com a interface `predict_proba` que o backend `sklearn` do DiCE espera."""

    def __init__(self, model, device) -> None:
        self.model, self.device = model, device

    def predict_proba(self, frame):
        x = torch.from_numpy(np.asarray(frame, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            return torch.softmax(self.model(x), 1).cpu().numpy()

    def predict(self, frame):
        return self.predict_proba(frame).argmax(1)


def build_dice(model, X_tr: np.ndarray, y_tr: np.ndarray, device, seed: int):
    """(explicador, nomes das colunas). Exige `pip install dice-ml`."""
    import dice_ml
    import pandas as pd

    cols = [f"x{f}" for f in range(X_tr.shape[1])]
    df = pd.DataFrame(X_tr, columns=cols).assign(y=y_tr)
    data = dice_ml.Data(dataframe=df, continuous_features=cols, outcome_name="y")
    m = dice_ml.Model(model=_Wrapped(model, device), backend="sklearn", model_type="classifier")
    np.random.seed(seed)
    return dice_ml.Dice(data, m, method="random"), cols


def dice_scores(dice, xi: np.ndarray, cj: int, total: int, rng, xj: np.ndarray | None = None) -> np.ndarray:
    """Fracao dos contrafactuais do DiCE que alteram cada atributo (zeros se nenhum foi achado).

    Com `xj`, os atributos em que x_i e x_j ja coincidem vao para o fim do ranking (pontuacao -1): o
    DiCE pode altera-los, mas leva-los ao valor de j nao muda nada no teste de trocas, e as demais
    tecnicas contrastivas e a ordem aleatoria tambem so ordenam os atributos que diferem.
    """
    import pandas as pd

    explainer, cols = dice
    query = pd.DataFrame([np.asarray(xi, dtype=float)], columns=cols)
    try:
        cfs = explainer.generate_counterfactuals(query, total_CFs=total, desired_class=int(cj), verbose=False)
        found = cfs.cf_examples_list[0].final_cfs_df
    except Exception:  # o DiCE levanta quando nao acha contrafactual  # noqa: BLE001
        found = None
    freq = np.zeros(len(xi))
    if found is not None and len(found):
        freq = (np.abs(found[cols].to_numpy(dtype=float) - np.asarray(xi)[None]) > 1e-6).mean(0)
    score = freq + 1e-9 * rng.random(len(xi))
    if xj is not None:
        score = np.where(np.abs(np.asarray(xj) - np.asarray(xi)) > 1e-9, score, -1.0)
    return score


def dice_counterfactuals(dice, xi: np.ndarray, cj: int, total: int, vary: list[int] | None = None) -> np.ndarray:
    """Os contrafactuais que o DiCE gera para x_i na classe c_j: (k, F), vazio se nenhum foi achado.

    `vary` restringe os atributos que o DiCE pode alterar (os demais sao imutaveis).
    """
    import pandas as pd

    explainer, cols = dice
    query = pd.DataFrame([np.asarray(xi, dtype=float)], columns=cols)
    kwargs = {} if vary is None else {"features_to_vary": [cols[f] for f in vary]}
    try:
        cfs = explainer.generate_counterfactuals(query, total_CFs=total, desired_class=int(cj), verbose=False,
                                                 **kwargs)
        found = cfs.cf_examples_list[0].final_cfs_df
    except Exception:  # o DiCE levanta quando nao acha contrafactual  # noqa: BLE001
        found = None
    if found is None or not len(found):
        return np.empty((0, len(xi)))
    return found[cols].to_numpy(dtype=float)


@torch.no_grad()
def minimal_flip(model, x_from, x_to, c_to: int, device, allowed=None, batch: int = 1 << 16) -> int | None:
    """Menor numero de atributos de `allowed` levados de x_from a x_to que faz o modelo prever c_to.

    Enumera os 2^|allowed| subconjuntos na GPU; None se nem todos juntos mudam a predicao.
    """
    allowed = np.arange(len(x_from)) if allowed is None else np.asarray(allowed)
    A = len(allowed)
    a = torch.as_tensor(np.asarray(x_from), dtype=torch.float32, device=device)
    b = torch.as_tensor(np.asarray(x_to), dtype=torch.float32, device=device)
    cols = torch.as_tensor(allowed, device=device)
    best = None
    for s in range(0, 1 << A, batch):
        masks = torch.arange(s, min(s + batch, 1 << A), device=device)
        bits = ((masks[:, None] >> torch.arange(A, device=device)) & 1).bool()
        rows = a.repeat(len(masks), 1)
        rows[:, cols] = torch.where(bits, b[cols], a[cols])
        hit = model(rows).argmax(1) == c_to
        if hit.any():
            k = int(bits[hit].sum(1).min())
            best = k if best is None else min(best, k)
    return best


def baseline_shapley(model, x_from, x_to, c_from: int, c_to: int, device, batch: int = 1 << 16):
    """Shapley exato de v(S) = m(x_to em S, x_from fora) − m(x_from), m = f_{c_to} − f_{c_from}.

    Devolve (φ, v(todos) − v(vazio)); a soma de φ e exatamente o segundo valor (eficiencia).
    """
    F = len(x_from)
    masks = torch.arange(1 << F, device=device)
    bits = ((masks[:, None] >> torch.arange(F, device=device)) & 1).bool()
    a = torch.as_tensor(x_from, dtype=torch.float32, device=device)
    b = torch.as_tensor(x_to, dtype=torch.float32, device=device)
    values = []
    with torch.no_grad():
        for s in range(0, len(bits), batch):
            logits = model(torch.where(bits[s : s + batch], b, a))
            values.append((logits[:, c_to] - logits[:, c_from]).double())
    v = torch.cat(values)
    size = bits.sum(1)
    w = torch.tensor([factorial(k) * factorial(F - k - 1) / factorial(F) for k in range(F)],
                     dtype=torch.float64, device=device)
    phi = np.empty(F)
    for f in range(F):
        without = masks[~bits[:, f]]
        phi[f] = float((w[size[without]] * (v[without | (1 << f)] - v[without])).sum())
    return phi, float(v[-1] - v[0])


def library_shap(model, X: np.ndarray, background: np.ndarray, device, max_evals: int, seed: int) -> np.ndarray:
    """SHAP intervencional da biblioteca para todos os logits: (n, F, K)."""
    import shap

    def f(x):
        with torch.no_grad():
            return model(torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)).cpu().numpy()
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.PermutationExplainer(f, masker, seed=seed)
    return explainer(X, max_evals=max_evals, silent=True).values


def logits_fn(model, device):
    def f(x):
        with torch.no_grad():
            return model(torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)).double().cpu().numpy()
    return f


def lime_contrastive(explainer: LimeTabularExplainer, model, xi, xj, ci: int, cj: int, device,
                     num_samples: int) -> np.ndarray:
    """LIME em regressao sobre a margem m = f_{c_j} − f_{c_i}, amostrando em torno de x_i.

    O modelo linear local da m ~ Σ_f w_f (x_f − x_{i,f})/σ_f; a previsao da mudanca de m ao levar
    cada atributo ao valor de j e w_f (x_{j,f} − x_{i,f})/σ_f, que e a pontuacao do atributo.
    """
    f = logits_fn(model, device)
    exp = explainer.explain_instance(xi, lambda X: f(X)[:, cj] - f(X)[:, ci], num_features=len(xi),
                                     num_samples=num_samples)
    w = np.zeros(len(xi))
    # no modo de regressao o LIME guarda em local_exp[1] os pesos e em local_exp[0] os mesmos
    # pesos com o sinal trocado (lime_tabular.py); a chave 1 e a do modelo linear ajustado
    for feat, weight in exp.local_exp[1]:
        w[feat] = weight
    return w * (xj - xi) / explainer.scaler.scale_


def lime_weights(explainer: LimeTabularExplainer, model, x, labels, device, num_samples: int) -> dict:
    """LIME de classificacao usual (atributos discretizados): pesos por classe pedida."""
    f = logits_fn(model, device)

    def proba(X):
        z = f(X)
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)
    exp = explainer.explain_instance(x, proba, labels=tuple(labels), num_features=len(x),
                                     num_samples=num_samples)
    out = {}
    for c in labels:
        w = np.zeros(len(x))
        for feat, weight in exp.local_exp[c]:
            w[feat] = weight
        out[c] = w
    return out


def margin_of(phi: np.ndarray, c_to: int, c_from: int) -> np.ndarray:
    """Atribuicao da diferenca de logits f_{c_to} − f_{c_from}, por linearidade do Shapley."""
    return phi[..., c_to] - phi[..., c_from]


def contrastive_margin(phi: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Atribuicao da margem um-contra-o-resto da classe c de cada amostra: (n, F)."""
    K = phi.shape[-1]
    own = phi[np.arange(len(c)), :, c]
    return own - (phi.sum(-1) - own) / (K - 1)


def sampled_shapley(model, x_from, x_to, c_from: int, c_to: int, device, budget: int, rng) -> np.ndarray:
    """Shapley de v(S) = m(x_to em S, x_from fora), por permutacoes: floor(budget/(F+1)) delas."""
    F = len(x_from)
    n_perm = max(1, budget // (F + 1))
    rows, perms = [], []
    for _ in range(n_perm):
        perm = rng.permutation(F)
        cur = x_from.copy()
        rows.append(cur.copy())
        for f in perm:
            cur[f] = x_to[f]
            rows.append(cur.copy())
        perms.append(perm)
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(rows, dtype=np.float32)).to(device))
        v = (logits[:, c_to] - logits[:, c_from]).double().cpu().numpy().reshape(n_perm, F + 1)
    phi = np.zeros(F)
    for k, perm in enumerate(perms):
        phi[perm] += np.diff(v[k])
    return phi / n_perm

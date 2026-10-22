"""Atribuicoes contrastivas de referencia para a leitura de pares (P6).

  integrated_gradients  gradientes integrados (Sundararajan et al., 2017) da margem
                        m = f_{c_j} − f_{c_i} no caminho reto de x_i a x_j. Pela completude,
                        Σ_f φ_f = m(x_j) − m(x_i): e a mesma quantidade que o Shapley com j de
                        referencia decompoe, por outro caminho (gradiente em vez de coalizoes).
  dice                  DiCE (Mothilal et al., 2020), metodo `random`, que so precisa de
                        `predict_proba`: gera contrafactuais de x_i para a classe de j, e a
                        pontuacao de cada atributo e a fracao dos contrafactuais que o alteram.
                        Os empates (frequencias iguais) sao desfeitos ao acaso, nunca pelo indice.
"""

from __future__ import annotations

import numpy as np
import torch


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


def dice_scores(dice, xi: np.ndarray, cj: int, total: int, rng) -> np.ndarray:
    """Fracao dos contrafactuais do DiCE que alteram cada atributo (zeros se nenhum foi achado)."""
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
    return freq + 1e-9 * rng.random(len(xi))

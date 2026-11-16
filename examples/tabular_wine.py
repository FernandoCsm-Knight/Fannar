"""Exemplo tabular: um MLP em PyTorch sobre a base `wine` do scikit-learn, com pandas.

    pip install "fannar[all]"
    python tabular_wine.py

Mostra `fannar.tabular.TabularInterpreter` recebendo um DataFrame (os nomes das colunas viram os
nomes dos atributos) e rótulos categóricos (as categorias viram os nomes das classes), e as
explicações de pares e de erros. Grava as figuras em ./fannar_wine/.
"""

import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.datasets import load_wine  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from fannar.tabular import TabularInterpreter  # noqa: E402

out = Path("fannar_wine")
out.mkdir(exist_ok=True)

# 1. dados: um DataFrame e rótulos categóricos com nomes
data = load_wine(as_frame=True)
X = data.data                                                   # 178 x 13, colunas com nome
y = pd.Series(pd.Categorical.from_codes(data.target, ["barolo", "grignolino", "barbera"]))
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
mean, std = X_tr.mean(), X_tr.std()
X_tr, X_te = (X_tr - mean) / std, (X_te - mean) / std          # o modelo treina na escala padronizada

# 2. modelo: guarde a cópia não treinada (o piso) antes de treinar
torch.manual_seed(0)
model = nn.Sequential()
model.add_module("block1", nn.Sequential(nn.Linear(13, 32), nn.ReLU()))
model.add_module("block2", nn.Sequential(nn.Linear(32, 32), nn.ReLU()))
model.add_module("head", nn.Linear(32, 3))
initial = copy.deepcopy(model)

xt = torch.tensor(X_tr.to_numpy(), dtype=torch.float32)
yt = torch.tensor(y_tr.cat.codes.to_numpy(), dtype=torch.long)
opt = torch.optim.Adam(model.parameters(), 1e-3)
for _ in range(300):
    opt.zero_grad()
    nn.functional.cross_entropy(model(xt), yt).backward()
    opt.step()
model.eval()

# 3. o interpretador: DataFrame e Series direto, sem converter nada
interp = TabularInterpreter(model, X_te, y_te, layers=["block1", "block2"], reference_model=initial)
print("atributos:", interp.feature_names[:4], "...")
print("classes:  ", interp.class_names)

report = interp.evaluate()
print(f"\nacurácia no teste: {report.meta['accuracy']:.3f}")
print(report.to_markdown())
interp.plot_report(report).savefig(out / "report.png", dpi=150)
interp.plot_embedding().savefig(out / "embedding.png", dpi=150)

# 4. um par: o que separa o vinho 0 do vizinho mais próximo com outra predição?
expl = interp.explain_pair(0)
names = interp.class_names
print(f"\npar ({expl.i}, {expl.j}): {names[expl.pred_i]} x {names[expl.pred_j]}")
for feat, vi, vj, share in expl.top(4):
    print(f"  {feat:30s} i = {vi:+.2f}  j = {vj:+.2f}  parcela {share:.0%}")
expl.plot().savefig(out / "pair.png", dpi=150)

# 5. os erros: o que mantém cada erro longe da sua classe verdadeira?
errs = interp.explain_errors()
print(f"\n{len(errs.errors)} erros; confusões (verdadeira -> predita):")
for t, p, n in errs.confusions():
    print(f"  {names[t]} -> {names[p]}: {n}")
ref = errs.reference_shares()
for feat, share, first in errs.top(4):
    f = interp.feature_names.index(feat)
    print(f"  {feat:30s} parcela nos erros {share:.0%} (em geral {ref[f]:.0%}), primeiro em {first:.0%}")
errs.plot().savefig(out / "errors.png", dpi=150)

# 6. validação por trocas: trocar os atributos na ordem do método muda a predição mais cedo?
print("\nteste de trocas (área, maior = melhor):", interp.swap_test(20))

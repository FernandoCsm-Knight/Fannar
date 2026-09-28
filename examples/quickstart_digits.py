"""Exemplo completo: um MLP em PyTorch sobre a base `digits` do scikit-learn (vem com o pacote).

    pip install "fannar[all]"
    python quickstart_digits.py

Grava o relatório (markdown e json) e as figuras em ./fannar_digits/.
"""

import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import fannar as fa  # noqa: E402

out = Path("fannar_digits")
out.mkdir(exist_ok=True)

X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

torch.manual_seed(0)
model = nn.Sequential()
model.add_module("block1", nn.Sequential(nn.Linear(64, 128), nn.ReLU()))
model.add_module("block2", nn.Sequential(nn.Linear(128, 128), nn.ReLU()))
model.add_module("block3", nn.Sequential(nn.Linear(128, 64), nn.ReLU()))
model.add_module("head", nn.Linear(64, 10))
initial = copy.deepcopy(model)  # o piso: a mesma rede antes do treino

opt = torch.optim.Adam(model.parameters(), 1e-3)
xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
for epoch in range(60):
    for idx in torch.randperm(len(xt)).split(64):
        opt.zero_grad()
        nn.functional.cross_entropy(model(xt[idx]), yt[idx]).backward()
        opt.step()
model.eval()

interp = fa.Interpreter(model, X_te, y_te, layers=["block1", "block2", "block3"], reference_model=initial,
                        feature_names=[f"pixel{t}" for t in range(64)])
report = interp.evaluate()
print(f"acurácia no teste: {report.meta['accuracy']:.3f}")
print(report.to_markdown())
report.to_json(out / "report.json")

interp.plot_report(report).savefig(out / "report.png", dpi=150)
interp.plot_embedding().savefig(out / "embedding.png", dpi=150)
interp.plot_gram("block2").savefig(out / "gram_block2.png", dpi=150)
interp.plot_spectrum("block2").savefig(out / "spectrum_block2.png", dpi=150)

expl = interp.explain_pair(0)
print(f"\npar ({expl.i}, {expl.j}): predições {expl.pred_i} e {expl.pred_j}")
for name, vi, vj, share in expl.top(5):
    print(f"  {name:8s} i = {vi:.2f}  j = {vj:.2f}  parcela {share:.0%}")
expl.plot().savefig(out / "pair.png", dpi=150)
print("\nteste de trocas (área, maior = melhor):", interp.swap_test(30, layer="block3"))

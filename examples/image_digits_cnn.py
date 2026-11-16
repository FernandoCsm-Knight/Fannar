"""Exemplo de imagens: uma CNN pequena sobre os dígitos 8x8 do scikit-learn, lidos como imagens.

    pip install "fannar[all]"
    python image_digits_cnn.py

Mostra `fannar.image.ImageInterpreter` com um `DataLoader`, nomes de classes, o espaço de cada
camada e o mapa de energia de uma imagem (onde a energia do modelo se concentra). Grava as figuras
em ./fannar_digits_cnn/.
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
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from fannar.image import ImageInterpreter  # noqa: E402

out = Path("fannar_digits_cnn")
out.mkdir(exist_ok=True)

# 1. dados como imagens (N, 1, 8, 8)
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32).reshape(-1, 1, 8, 8)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# 2. uma CNN com blocos nomeados: os nomes são o que o Fannar sonda
torch.manual_seed(0)
model = nn.Sequential()
model.add_module("conv1", nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU()))
model.add_module("conv2", nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU()))
model.add_module("conv3", nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()))
model.add_module("head", nn.Sequential(nn.Flatten(), nn.Linear(32 * 8 * 8, 10)))
initial = copy.deepcopy(model)

train = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=64, shuffle=True)
opt = torch.optim.Adam(model.parameters(), 1e-3)
for epoch in range(20):
    for xb, yb in train:
        opt.zero_grad()
        nn.functional.cross_entropy(model(xb), yb).backward()
        opt.step()
model.eval()

# 3. o interpretador recebe o DataLoader de teste como está: (imagem, rótulo) por lote
test = DataLoader(TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)), batch_size=128)
interp = ImageInterpreter(model, test, y_te, layers=["conv1", "conv2", "conv3"], reference_model=initial,
                          class_names=[str(d) for d in range(10)])

report = interp.evaluate()
print(f"acurácia no teste: {report.meta['accuracy']:.3f}")
print(report.to_markdown())
print("\nformato das fontes de conv3 (objetos, canais, posições):",
      interp.sources.layers["conv3"]["activations"].shape)     # grid=4 -> 16 posições

interp.plot_report(report).savefig(out / "report.png", dpi=150)
interp.plot_embedding().savefig(out / "embedding.png", dpi=150)
interp.plot_spectrum("conv3").savefig(out / "spectrum_conv3.png", dpi=150)

# 4. dentro de uma imagem: as posições viram os objetos
image = X_te[0]
emap = interp.energy_map(image, "conv2")                        # (8, 8): energia retida por posição
print(f"\nimagem 0 (dígito {y_te[0]}): energia por posição em conv2")
print(np.array2string(emap / emap.max(), precision=1, suppress_small=True))
ink = image[0] > 0.25
print(f"fração da energia sobre o traço do dígito: {emap[ink].sum() / emap.sum():.0%} "
      f"(o traço cobre {ink.mean():.0%} da imagem)")
interp.plot_energy_map(image, "conv2", image=image[0]).savefig(out / "energy_conv2.png", dpi=150)

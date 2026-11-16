# Quick start

This page interprets a small PyTorch network trained on scikit-learn's `digits` dataset, layer by
layer. The full script is
[`examples/quickstart_digits.py`](https://github.com/FernandoCsm-Knight/Fannar/blob/main/examples/quickstart_digits.py).

```bash
pip install "fannar[all]"
```

## 1. Keep a copy of the untrained model

Every reading needs a reference to be meaningful. The natural one is the same network before
training, so copy it before the first optimisation step:

```python
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

torch.manual_seed(0)
model = nn.Sequential()
model.add_module("block1", nn.Sequential(nn.Linear(64, 128), nn.ReLU()))
model.add_module("block2", nn.Sequential(nn.Linear(128, 128), nn.ReLU()))
model.add_module("block3", nn.Sequential(nn.Linear(128, 64), nn.ReLU()))
model.add_module("head", nn.Linear(64, 10))
initial = copy.deepcopy(model)            # the floor: the same network before training
```

## 2. Train as usual

```python
opt = torch.optim.Adam(model.parameters(), 1e-3)
xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
for epoch in range(60):
    for idx in torch.randperm(len(xt)).split(64):
        opt.zero_grad()
        nn.functional.cross_entropy(model(xt[idx]), yt[idx]).backward()
        opt.step()
model.eval()
```

## 3. Build the interpreter

Each 64-pixel image is a row of 64 features here, so the entry point is
[`fannar.tabular`](guides/tabular.md):

```python
import fannar as fa
from fannar.tabular import TabularInterpreter

interp = TabularInterpreter(
    model, X_te, y_te,                        # the trained model and the objects to interpret
    layers=["block1", "block2", "block3"],    # names from model.named_modules()
    reference_model=initial,                  # optional, but recommended: the floor
    feature_names=[f"pixel{t}" for t in range(64)],
)
```

`TabularInterpreter` is an `fa.Interpreter` with defaults for tables. `fa.Interpreter` takes the
same arguments and gives the same numbers. For images, token sequences and trees, see
[Choosing a module](guides/other-models.md).

Nothing is computed yet. The first call that needs the spaces runs `interp.fit()`, which
extracts the sources of every layer in one pass over `X_te` and builds one
[`GramSpace`](guides/gram-space.md) per layer.

## 4. Read the report

```python
report = interp.evaluate()
print(f"test accuracy: {report.meta['accuracy']:.3f}")
print(report.to_markdown())               # also report.to_frame() and report.to_json("r.json")
interp.plot_report(report)                # readings × layers
```

For each layer the report holds the readings of three representations:

- `joint`: the method, with the three sources combined;
- `activations`: a baseline with the activations alone;
- `floor`: the same pipeline on the untrained network, measured against the trained network's
  behaviour.

A reading only says something about training when `joint` is above `floor`. What each reading
asks is described in [Readings and reports](guides/readings.md).

## 5. Look at the spaces

```python
interp.plot_embedding()                   # each layer on its two dominant modes
interp.plot_gram("block2")                # component Grams, Hadamard product, final Gram
interp.plot_spectrum("block2")            # eigenvalues and participation rank

space = interp.space("block2")            # the GramSpace itself
space.neighbors(0, 5)
space.energies()["E_ratio"]
```

## 6. Explain decisions

```python
expl = interp.explain_pair(0)             # object 0 vs. its nearest object with another prediction
for name, vi, vj, share in expl.top(5):
    print(f"{name:8s} i = {vi:.2f}  j = {vj:.2f}  share {share:.0%}")
expl.plot()

errs = interp.explain_errors()            # which features keep the errors away from their class
errs.top(5)
errs.plot()

interp.swap_test(30, layer="block3")      # validation outside the geometry
```

See [Explanations](guides/explanations.md) for what these numbers mean and how they are
validated.

## Saving figures

Every `plot_*` method returns a Matplotlib figure:

```python
interp.plot_report(report).savefig("report.png", dpi=150)
```

## Next steps

- [Examples](examples.md): one runnable script per kind of data, with its output.
- [The method](concepts.md): what the three sources are and why the defaults are what they are.
- [Choosing a module](guides/other-models.md): `fannar.tabular`, `fannar.image`, `fannar.text`,
  `fannar.tree`, and any model through a function.

# Installation

Fannar needs Python 3.10 or later. The core depends only on NumPy and Matplotlib; the model
integrations are optional extras.

```bash
pip install fannar
```

## Optional dependencies

| extra | installs | needed for |
|---|---|---|
| `torch` | `torch>=2.0` | interpreting PyTorch models (`TorchSources`, energy maps) |
| `sklearn` | `scikit-learn>=1.2` | interpreting scikit-learn decision trees (`TreeSources`) |
| `pandas` | `pandas>=1.5` | `Report.to_frame()` and the `to_frame()` of the explanations |
| `all` | all of the above | |

```bash
pip install "fannar[torch]"
pip install "fannar[all]"
```

`fannar.TorchSources` is `None` when PyTorch is not installed; the rest of the library works
without it.

## From source

```bash
git clone https://github.com/FernandoCsm-Knight/Fannar.git
cd Fannar
pip install -e ".[test]"
pytest
```

To build this documentation locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Checking the install

```python
import numpy as np
import fannar as fa

X = np.random.default_rng(0).standard_normal((50, 8))
space = fa.KernelPipeline(["features"])(sources={"features": X})
print(space, space.is_psd(), space.participation_rank())
```

"""CIFAR-100 (Krizhevsky, 2009): 100 classes finas agrupadas em 20 superclasses.

A segunda base de imagens do artigo. Ela entra por dois motivos: e a base em que a
literatura de similaridade de representacoes mais se apoia (Kornblith et al., 2019), e a
hierarquia oficial de superclasses e uma referencia semantica externa ao modelo -- o modelo e
treinado so nas 100 classes finas e nunca ve a superclasse, que serve apenas para conferir se
as distancias entre classes refletem a semantica (pergunta P9).

A base fica fora do repositorio, em `~/Desktop/torch/data/cifar100/cifar-100-python/` (os
pickles oficiais `train`, `test` e `meta`). A particao e a oficial: 50 000 de treino e 10 000
de teste. O dicionario devolvido tem as mesmas chaves que `data_pets.load_pets` usa
(`images`, `labels`, `split`, `classes`) mais `coarse` e `coarse_classes`.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

ROOT = Path.home() / "Desktop" / "torch" / "data" / "cifar100" / "cifar-100-python"
SIZE = 32


def _unpickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def load_cifar100(root: Path = ROOT, cache: Path | None = None) -> dict:
    cache = cache or Path("outputs") / "cifar100_32.npz"
    if cache.exists():
        blob = np.load(cache)
        return {k: blob[k] for k in blob.files}
    parts = [_unpickle(root / name) for name in ("train", "test")]
    meta = _unpickle(root / "meta")
    images = np.concatenate([p["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) for p in parts])
    out = {
        "images": images.astype(np.uint8),
        "labels": np.concatenate([np.asarray(p["fine_labels"]) for p in parts]).astype(np.int64),
        "coarse": np.concatenate([np.asarray(p["coarse_labels"]) for p in parts]).astype(np.int64),
        "split": np.concatenate([np.full(len(p["fine_labels"]), s) for s, p in enumerate(parts)]).astype(np.int64),
        "classes": np.asarray(meta["fine_label_names"]),
        "coarse_classes": np.asarray(meta["coarse_label_names"]),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out

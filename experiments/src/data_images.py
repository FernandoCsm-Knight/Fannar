"""Ponto unico de acesso as bases de imagens: Oxford-IIIT Pet e CIFAR-100.

`load_images(nome)` devolve o dicionario da base (`images`, `labels`, `split`, `classes`, ...)
e `hierarchy(nome, dados)` o grupo semantico de cada classe fina, que nunca entra no treino:

  pets      a especie de cada raca (0 = gato, 1 = cachorro), da lista oficial da base;
  cifar100  a superclasse oficial de cada classe (20 grupos de 5).
"""

from __future__ import annotations

import numpy as np

DATASETS = ("pets", "cifar100")


def load_images(name: str) -> dict:
    if name == "pets":
        from .data_pets import SIZE, load_pets
        return load_pets(SIZE)
    if name == "cifar100":
        from .data_cifar import load_cifar100
        return load_cifar100()
    raise KeyError(name)


def hierarchy(name: str, data: dict) -> np.ndarray:
    """Grupo semantico de cada classe fina, (K,)."""
    group = data["species"] if name == "pets" else data["coarse"]
    labels = data["labels"]
    out = np.full(len(data["classes"]), -1, dtype=np.int64)
    for c in range(len(out)):
        values = np.unique(group[labels == c])
        assert len(values) == 1, f"classe {c} em mais de um grupo"
        out[c] = values[0]
    return out


def same_group(groups: np.ndarray) -> np.ndarray:
    """(K, K) booleana: as duas classes estao no mesmo grupo semantico."""
    return groups[:, None] == groups[None, :]

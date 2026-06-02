"""Banco de conceitos como vetores no espaço de representação."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ConceptVector:
    """Um conceito representado por um vetor no espaço de representação."""

    name: str
    vector: torch.Tensor  # (d,)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vector.ndim != 1:
            self.vector = self.vector.reshape(-1)


class ConceptBank:
    """Coleção de conceitos com a mesma dimensionalidade ``d``."""

    def __init__(self) -> None:
        self._concepts: dict[str, ConceptVector] = {}

    def add(self, name: str, vector: torch.Tensor, **metadata) -> None:
        cv = ConceptVector(name=name, vector=vector, metadata=metadata)
        if self._concepts:
            d = next(iter(self._concepts.values())).vector.shape[0]
            if cv.vector.shape[0] != d:
                raise ValueError(
                    f"Conceito '{name}' tem d={cv.vector.shape[0]}, esperado d={d}."
                )
        self._concepts[name] = cv

    def names(self) -> list[str]:
        return list(self._concepts.keys())

    def matrix(self) -> torch.Tensor:
        """Empilha os conceitos em ``(num_concepts, d)``."""
        if not self._concepts:
            raise ValueError("ConceptBank vazio.")
        return torch.stack([c.vector for c in self._concepts.values()])

    def __len__(self) -> int:
        return len(self._concepts)

    def __getitem__(self, name: str) -> ConceptVector:
        return self._concepts[name]

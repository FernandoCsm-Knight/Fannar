"""Utilidades para construir vetores de conceito a partir de exemplos."""

from __future__ import annotations

import torch


def concept_from_examples(positives: torch.Tensor, negatives: torch.Tensor | None = None) -> torch.Tensor:
    """Vetor de conceito como diferença de médias (estilo CAV simplificado).

    ``positives`` ``(p, d)``, ``negatives`` ``(m, d)`` ou ``None``.
    Retorna um vetor ``(d,)``. Sem negativos, usa apenas a média dos positivos.
    """
    pos_mean = positives.mean(dim=0)
    v = pos_mean if negatives is None else pos_mean - negatives.mean(dim=0)
    return v

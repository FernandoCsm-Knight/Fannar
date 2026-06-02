"""Estabilidade local: distância média aos k vizinhos mais próximos."""

from __future__ import annotations

import torch

from ..types import DiagnosticResult


def stability(D: torch.Tensor, k: int = 5) -> DiagnosticResult:
    """``Stab(i) = mean_{j in Nk(i)} d(i, j)`` para cada objeto ``i``.

    Retorna um vetor ``(n,)`` em ``detail['per_item']`` e a média global em
    ``value``. Exclui a auto-distância (diagonal).
    """
    n = D.shape[0]
    k = min(k, n - 1) if n > 1 else 0
    if k <= 0:
        return DiagnosticResult(
            name="stability", value=D.new_tensor(0.0), available=n > 1,
            detail={"per_item": D.new_zeros(n), "k": 0},
        )
    Dmod = D.clone()
    Dmod.fill_diagonal_(float("inf"))
    # k menores por linha
    knn_vals, _ = torch.topk(Dmod, k, dim=1, largest=False)
    per_item = knn_vals.mean(dim=1)
    return DiagnosticResult(
        name="stability",
        value=per_item.mean(),
        detail={"per_item": per_item, "k": k},
    )

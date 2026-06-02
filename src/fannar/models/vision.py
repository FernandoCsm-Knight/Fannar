"""Conveniências para modelos de visão: extração de patches.

LIMITAÇÃO: ``patchify`` assume imagem ``(C, H, W)`` ou ``(N, C, H, W)`` com
``H``, ``W`` divisíveis por ``patch_size``. Resíduos de borda são descartados.
"""

from __future__ import annotations

import torch


def patchify(image: torch.Tensor, patch_size: int = 16) -> tuple[torch.Tensor, tuple[int, int]]:
    """Divide a imagem em patches achatados.

    Retorna ``(patches, grid)`` onde ``patches`` tem shape ``(num_patches, d)``
    com ``d = C * patch_size * patch_size`` e ``grid = (n_h, n_w)``.
    Para batch ``(N, C, H, W)`` usa a primeira amostra.
    """
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Esperado (C,H,W) ou (N,C,H,W), recebido {tuple(image.shape)}.")
    c, h, w = image.shape
    n_h, n_w = h // patch_size, w // patch_size
    img = image[:, : n_h * patch_size, : n_w * patch_size]
    # (C, n_h, ps, n_w, ps) -> (n_h*n_w, C*ps*ps)
    patches = img.reshape(c, n_h, patch_size, n_w, patch_size)
    patches = patches.permute(1, 3, 0, 2, 4).reshape(n_h * n_w, c * patch_size * patch_size)
    return patches, (n_h, n_w)


def patch_grid_labels(grid: tuple[int, int]) -> list[str]:
    n_h, n_w = grid
    return [f"p{r}_{cc}" for r in range(n_h) for cc in range(n_w)]

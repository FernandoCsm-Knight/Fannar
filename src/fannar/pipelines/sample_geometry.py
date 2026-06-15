"""Geometria entre amostras (textos/frases): kernel RKHS K_W ⊙ K_A ⊙ K_G por camada.

Cada amostra é representada por suas ativações e gradientes médios sobre a
sequência de tokens.  O kernel entre amostras é o produto de Hadamard:

    K_tipos = K_W ⊙ K_A ⊙ K_G

onde K_W usa gram_W ⊗ gram_W como tensor de métrica (Mahalanobis), K_A e K_G
são kernels gaussianos euclidianos sobre ativações e gradientes, respectivamente.
"""

from __future__ import annotations

import torch

import torch

from ..gram.distance import pairwise_sq_dists as _pairwise_sq_dists


def pool_sequence(x: torch.Tensor) -> torch.Tensor:
    """Média temporal (T, d) → (d,). Tensores 1-D passam inalterados."""
    if x.ndim == 1:
        return x
    return x.mean(dim=0)


def _gaussian_kernel_from_d2(d2: torch.Tensor) -> torch.Tensor:
    """Kernel gaussiano a partir de distâncias quadradas com heurística da mediana."""
    off = d2.flatten()
    positive = off[off > 0]
    med = positive.median() if positive.numel() > 0 else d2.new_tensor(1.0)
    gamma = 1.0 / (2.0 * med.clamp_min(1e-12))
    return torch.exp(-gamma * d2)


def build_sample_kernel(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    gram_W: torch.Tensor,
) -> torch.Tensor:
    """Kernel RKHS entre N amostras para uma única camada.

    Parameters
    ----------
    activations : (N, d)  ativações médias (pooled sobre tokens) por amostra
    gradients   : (N, d)  gradientes médios (pooled sobre tokens) por amostra
    gram_W      : (d, d)  matriz de Gram dos pesos da camada

    Returns
    -------
    K_tipos : (N, N)  K_W ⊙ K_A ⊙ K_G  (produto de Hadamard)

    Notes
    -----
    K_W usa gram_W @ gram_W como tensor de métrica Mahalanobis, de modo que
    K_tipos = K_W ⊙ K_A ⊙ K_G é PSD pelo teorema de Moore–Aronszajn.
    """
    M = gram_W @ gram_W
    K_W = _gaussian_kernel_from_d2(_pairwise_sq_dists(activations, metric=M))
    K_A = _gaussian_kernel_from_d2(_pairwise_sq_dists(activations))
    K_G = _gaussian_kernel_from_d2(_pairwise_sq_dists(gradients))
    return K_W * K_A * K_G


def class_pair_means(
    D: torch.Tensor,
    categories: list[str],
) -> dict[tuple[str, str], float]:
    """Média das distâncias por par de categorias.

    Para pares distintos (A, B) retorna a média de todas as entradas D[i,j]
    com categoria(i)=A e categoria(j)=B.  Para o par (A, A) retorna a média
    das entradas fora da diagonal dentro da classe.

    Parameters
    ----------
    D          : (N, N) matriz de distâncias entre amostras.
    categories : lista de N strings com a categoria de cada amostra.

    Returns
    -------
    Dicionário ``{(cat_a, cat_b): média}`` para todos os pares não-ordenados.
    """
    unique = sorted(set(categories))
    idx = {c: [i for i, ci in enumerate(categories) if ci == c] for c in unique}
    result: dict[tuple[str, str], float] = {}
    for i, ca in enumerate(unique):
        for cb in unique[i:]:
            ia = torch.tensor(idx[ca], device=D.device)
            ib = torch.tensor(idx[cb], device=D.device)
            sub = D[ia][:, ib]
            if ca == cb:
                n = sub.shape[0]
                if n < 2:
                    result[(ca, cb)] = 0.0
                    continue
                off = sub[~torch.eye(n, dtype=torch.bool, device=D.device)]
                result[(ca, cb)] = float(off.mean())
            else:
                result[(ca, cb)] = float(sub.mean())
    return result

"""Gram cruzada de duas imagens: as posicoes das duas no mesmo espaco.

Por que o indice muda tudo. Nos conjuntos de nivel os objetos sao as posicoes de *uma*
imagem, e por isso uma direcao do espaco e um padrao espacial -- algo que se desenha. Na
Gram de imagens os objetos sao imagens, e uma direcao e uma distribuicao sobre a amostra,
que so ganha sentido por exemplares. Para perguntar "quais caracteristicas diferenciam
estas duas entradas" e obter uma figura, o indice precisa ser a posicao, e as duas imagens
precisam estar na *mesma* Gram.

Construcao: as P posicoes de A e as P de B viram 2P objetos, com as mesmas tres fontes da
eq. `fonte_parametros` e a mesma composicao (Hadamard das componentes lineares, depois
T_tr∘T_c). As distancias sao lidas apos a normalizacao angular, com piso na diagonal, que e
a variante que separa padrao de intensidade.

Tres leituras:

  correspondencia     para cada posicao de A, a distancia a posicao mais proxima de B.
                      Baixa = "isto tem equivalente la"; alta = caracteristica de A que o
                      modelo nao encontra em B, isto e, o que diferencia as duas.
  nivel cruzado       as faixas de nivel de uma referencia de A, desenhadas sobre as duas
                      imagens: onde em B mora o mesmo conceito.
  energia conjunta    a energia por posicao no espaco conjunto, quando pedida.

Uma decisao que precisa ser reportada: Γ depende da classe-alvo da margem. Com cada imagem
usando a sua propria predicao, "sem correspondente" mistura diferenca de aparencia com
diferenca de decisao; o controle e repetir com um alvo comum as duas (`target_mode`
`fixed`).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .pixel_energy import _unit_scale, joint_transform


def _sources(acts: torch.Tensor, grads: torch.Tensor, max_side: int | None):
    if max_side and acts.shape[-1] > max_side:
        acts = F.adaptive_avg_pool2d(acts, max_side)
        grads = F.adaptive_avg_pool2d(grads, max_side)
    h, w = acts.shape[-2:]
    return acts.flatten(2).transpose(1, 2), grads.flatten(2).transpose(1, 2), h, w


def cross_gram(
    acts_a: torch.Tensor,
    grads_a: torch.Tensor,
    acts_b: torch.Tensor,
    grads_b: torch.Tensor,
    metric: torch.Tensor,
    device: torch.device,
    max_side: int | None = 32,
    joint: str = "trace_center",
    beta: float = 1.0,
    floor: float = 0.1,
) -> tuple[np.ndarray, int, int, int]:
    """Gram (2P, 2P) das posicoes de A seguidas das de B, ja transformada e normalizada."""
    a_flat, ga_flat, h, w = _sources(acts_a, grads_a, max_side)
    b_flat, gb_flat, _, _ = _sources(acts_b, grads_b, max_side)
    A = torch.cat([a_flat, b_flat], dim=1).to(device, torch.float64)  # (1, 2P, C)
    G = torch.cat([ga_flat, gb_flat], dim=1).to(device, torch.float64)
    M = metric.to(device, torch.float64)
    comps = [
        _unit_scale(A @ M @ A.transpose(1, 2)),
        _unit_scale(A @ A.transpose(1, 2)),
        _unit_scale(G @ G.transpose(1, 2)),
    ]
    joint_gram, _ = joint_transform(comps[0] * comps[1] * comps[2], joint)
    g = joint_gram[0].cpu().numpy()
    if beta:
        d = np.diag(g)
        scale = np.maximum(d, floor * max(float(np.median(d)), 1e-300)) ** (beta / 2)
        g = g / np.outer(scale, scale)
    return g, h * w, h, w


def distance_matrix(g: np.ndarray) -> np.ndarray:
    d = np.diag(g)
    return np.sqrt(np.maximum(d[:, None] + d[None, :] - 2.0 * g, 0.0))


def correspondence(g: np.ndarray, p: int, h: int, w: int) -> dict:
    """Distancia de cada posicao a mais proxima da *outra* imagem, e o seu correspondente."""
    d = distance_matrix(g)
    cross = d[:p, p:]
    return {
        "no_counterpart_a": cross.min(1).reshape(h, w),
        "no_counterpart_b": cross.min(0).reshape(h, w),
        "match_a": cross.argmin(1).reshape(h, w),
        "match_b": cross.argmin(0).reshape(h, w),
        "distance": d,
    }


def level_from(d: np.ndarray, s0: int, p: int, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Funcao distancia a partir de s0, separada nas duas imagens."""
    return d[s0, :p].reshape(h, w), d[s0, p:].reshape(h, w)


def reference_position(no_counterpart: np.ndarray, energy: np.ndarray | None = None) -> int:
    """Referencia em A: a posicao com melhor correspondencia em B (o conceito compartilhado)."""
    flat = no_counterpart.reshape(-1)
    if energy is not None:
        # entre as posicoes bem correspondidas, a de maior energia
        keep = flat <= np.quantile(flat, 0.25)
        scores = np.where(keep, energy.reshape(-1), -np.inf)
        return int(scores.argmax())
    return int(flat.argmin())


def pooled_features(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Saida do pooling global, isto e, a entrada da camada linear."""
    h = x
    for name in model.block_names:
        h = model.get_block(name)(h)
    return model.head[1](model.head[0](h))


@torch.no_grad()
def separation_curves(
    model: torch.nn.Module,
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    field: np.ndarray,
    device: torch.device,
    steps: int = 20,
    seed: int = 0,
    classes: tuple[int, int] | None = None,
) -> dict:
    """Apaga posicoes de A na ordem de `field` e mede a separacao entre A e B.

    Com `classes = (c_a, c_b)` (as predicoes de A e de B), a leitura principal e a margem de
    decisao em A, logit[c_a] - logit[c_b], normalizada pela da imagem integra: e a propria
    pergunta "por que A foi para c_a e nao para c_b". A distancia entre caracteristicas nao
    responde a ela -- apagar qualquer parte do animal afasta A de B (A vira fundo), e o mapa
    que so marca o centro ganha dela tanto quanto o metodo.

    A separacao e a distancia de cosseno entre as caracteristicas do pooling global,
    limitada em [0, 2] e normalizada pelo valor com a imagem integra. A norma no espaco de
    logits fica como leitura secundaria: ela **nao** serve de metrica primaria porque, para
    um par que a rede prediz igual, a separacao inicial e quase nula e a razao explode.

    Se o mapa localiza o que diferencia as duas entradas, apagar primeiro as posicoes sem
    correspondente deve derrubar a separacao mais rapido do que apagar primeiro as posicoes
    com correspondente; `gap` e a diferenca entre as duas areas, ~0 para uma ordem aleatoria.
    """
    size = x_a.shape[-2:]
    m = torch.from_numpy(np.nan_to_num(field.astype(np.float32), nan=0.0))[None, None]
    up = F.interpolate(m, size=size, mode="bilinear", align_corners=False).flatten()
    gen = torch.Generator().manual_seed(seed)
    jitter = torch.rand(up.shape, generator=gen) * 1e-6 * up.abs().max().clamp_min(1e-30)
    order = torch.argsort(up + jitter, descending=True)
    npix = len(order)
    rank = torch.empty_like(order)
    rank.scatter_(0, order, torch.arange(npix))
    counts = torch.round(torch.linspace(0, 1, steps + 1) * npix).long()
    fractions = (counts.float() / npix).numpy()

    morf_mask = (rank[None, :] < counts[:, None]).reshape(steps + 1, 1, *size).to(device)
    lerf_mask = (rank[None, :] >= npix - counts[:, None]).reshape(steps + 1, 1, *size).to(device)
    xa = x_a.to(device)[None]
    xb = x_b.to(device)[None]
    batch = torch.cat([xa * ~morf_mask, xa * ~lerf_mask])
    feats = pooled_features(model, batch)
    feats_b = pooled_features(model, xb)
    logits = model.head[2](feats)
    logits_b = model.head[2](feats_b)

    cosine = 1.0 - torch.nn.functional.cosine_similarity(feats, feats_b, dim=1)
    logit_gap = (logits - logits_b).norm(dim=1)
    clean_cos, clean_logit = float(cosine[0]), float(logit_gap[0])
    cos = (cosine / max(clean_cos, 1e-12)).cpu().numpy()
    lg = (logit_gap / max(clean_logit, 1e-12)).cpu().numpy()

    def area(y):
        return float(((y[1:] + y[:-1]) * 0.5 * np.diff(fractions)).sum())

    morf, lerf = cos[: steps + 1], cos[steps + 1 :]
    margin = {}
    if classes is not None:
        ca, cb = classes
        m = (logits[:, ca] - logits[:, cb]).cpu().numpy()
        m = m / max(float(m[0]), 1e-12)
        margin = {"margin_morf": m[: steps + 1], "margin_lerf": m[steps + 1 :],
                  "gap_margin": area(m[steps + 1 :]) - area(m[: steps + 1]),
                  "clean_margin": float((logits[0, ca] - logits[0, cb]).item())}
    return margin | {
        "fractions": fractions,
        "morf": morf,
        "lerf": lerf,
        "auc_morf": area(morf),
        "auc_lerf": area(lerf),
        "gap": area(lerf) - area(morf),
        "gap_logits": area(lg[steps + 1 :]) - area(lg[: steps + 1]),
        "clean_separation": clean_cos,
        "clean_separation_logits": clean_logit,
    }

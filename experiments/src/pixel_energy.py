"""Energia espectral por pixel: o campo escalar do Comentario `rem:diagnostic`.

Aqui o objeto z_s e uma **posicao espacial** de uma unica imagem, na resolucao nativa
do bloco (32x32 em stem-b2, 16x16 em b3-b4, 8x8 em b5-b6). Cada posicao carrega as
mesmas tres fontes do artigo, lidas ao longo do eixo de canais:

  A  a_s in R^C                estado pos-ReLU do bloco
  Γ  g_s in R^C                derivada da margem contrastiva nesse estado
  W  p_s = sum_c a_s[c] W[c,:]  filtro efetivo ativado; <p_s, p_s'> = a_s^T (W W^T) a_s'

A Gram de cada fonte e P x P (P = H*W), a conjunta e o produto de Hadamard e a
transformacao final e a do artigo (T_tr o T_c). A energia de cada posicao nas direcoes
principais sai direto da decomposicao espectral, sem mapa auxiliar:

  Ê_s∥ = sum_{k<=r} λ̂_k V_sk²,     G_ss = sum_k λ̂_k V_sk²,

que e a eq. `empirical_energies` com os pixels no lugar das imagens; r pela razao de
participacao (`eq:r_participation`), por imagem.

Duas escolhas que mudam o que o campo mede
------------------------------------------
*Componentes.* O padrao e o nucleo linear sem normalizacao angular (`component="id"`).
A angular forca K_ss = 1 em toda posicao e apaga a magnitude: um pixel morto
(a_s ≈ 0) passaria a ter a mesma norma que um pixel ativo, e o campo mede justamente
essa norma. A escala global de cada componente nao importa -- ela sai do produto de
Hadamard como um escalar e o T_tr final a remove. `component="angular"` fica disponivel,
com `shift`, para comparar.

*Centramento.* Com T_tr o T_c, G_ss e a distancia da posicao ao centroide das posicoes
da imagem, entao o campo mede o quanto cada posicao se afasta do padrao medio da imagem
dentro do subespaco principal. Com `joint="trace"` (sem centramento) e a norma bruta, e
o modo dominante tende a ser o proprio padrao medio.

Degenerescencia conhecida
-------------------------
A cabeca da rede e pooling global + linear, entao em b6 o gradiente e o mesmo vetor
(w_{c*} - w̄)/HW em toda posicao: K_Γ = c*J tem posto 1. Na conjunta ele so multiplica
por uma constante, e o mapa de b6 e o de (W, A) -- **nao depende do alvo da margem**.
Sozinho, depois do centramento, K_Γ e nulo e o mapa de Γ em b6 e indefinido
(`degenerate`). `gamma_participation` (tr²/‖K_Γ‖_F², igual a 1 sse posto 1) mede isso
em todo bloco.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

REPS = ("joint", "activations", "params", "grads")


@dataclass
class PixelSources:
    """Estados e gradientes na resolucao nativa, por bloco."""

    blocks: list[str]
    acts: dict[str, torch.Tensor]  # bloco -> (N, C, H, W), cpu float32
    grads: dict[str, torch.Tensor]  # bloco -> (N, C, H, W), cpu float32
    metric: dict[str, torch.Tensor]  # bloco -> (C, C) = W W^T, float64
    labels: np.ndarray
    preds: np.ndarray
    logits: np.ndarray
    target: np.ndarray  # classe usada na margem


def margin_target(logits: torch.Tensor, mode: str) -> torch.Tensor:
    """`pred`: a classe predita (definicao da fonte). `second`: a vice, controle de alvo.

    `fixed` (ou `fixed:c`) usa a mesma classe para todas as imagens. E o controle necessario
    quando duas imagens sao comparadas entre si: com o alvo proprio de cada uma, o gradiente
    difere tambem por causa da decisao, e nao so da aparencia.
    """
    if mode == "pred":
        return logits.argmax(1)
    if mode == "second":
        return logits.topk(2, dim=1).indices[:, 1]
    if mode.startswith("fixed"):
        cls = int(mode.split(":")[1]) if ":" in mode else 0
        return torch.full((len(logits),), cls, device=logits.device, dtype=torch.long)
    raise ValueError(f"alvo desconhecido: {mode}")


def contrastive_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Logit do alvo menos a media dos demais (eq. `margem`)."""
    n, k = logits.shape
    rows = torch.arange(n, device=logits.device)
    others = (logits.sum(1) - logits[rows, target]) / (k - 1)
    return logits[rows, target] - others


def extract_pixel_sources(
    model: torch.nn.Module,
    x: torch.Tensor,
    labels: np.ndarray,
    blocks: list[str],
    device: torch.device,
    batch: int = 50,
    target: str = "pred",
) -> PixelSources:
    """Um forward e um backward por lote, com hook na saida de cada bloco, sem pooling."""
    store: dict[str, torch.Tensor] = {}

    def hook(name):
        def fn(_mod, _inputs, output):
            output.retain_grad()
            store[name] = output

        return fn

    handles = [model.get_block(b).register_forward_hook(hook(b)) for b in blocks]
    acts: dict[str, list[torch.Tensor]] = {b: [] for b in blocks}
    grads: dict[str, list[torch.Tensor]] = {b: [] for b in blocks}
    preds, logits_all, targets = [], [], []
    model.eval()
    try:
        for start in range(0, len(x), batch):
            xb = x[start : start + batch].to(device)
            store.clear()
            logits = model(xb)
            t = margin_target(logits.detach(), target)
            model.zero_grad(set_to_none=True)
            contrastive_margin(logits, t).sum().backward()
            for b in blocks:
                acts[b].append(store[b].detach().float().cpu())
                grads[b].append(store[b].grad.detach().float().cpu())
            preds.append(logits.argmax(1).cpu().numpy())
            logits_all.append(logits.detach().cpu().numpy())
            targets.append(t.cpu().numpy())
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)

    metric = {}
    for b in blocks:
        conv = model.get_block(b).out_conv
        w = conv.weight.detach().double().reshape(conv.out_channels, -1)
        metric[b] = (w @ w.T).cpu()

    return PixelSources(
        blocks=list(blocks),
        acts={b: torch.cat(v) for b, v in acts.items()},
        grads={b: torch.cat(v) for b, v in grads.items()},
        metric=metric,
        labels=np.asarray(labels),
        preds=np.concatenate(preds),
        logits=np.concatenate(logits_all),
        target=np.concatenate(targets),
    )


# --------------------------------------------------------------------------
# Grams P x P, em lote de imagens
# --------------------------------------------------------------------------


def _diag(K: torch.Tensor) -> torch.Tensor:
    return torch.diagonal(K, dim1=1, dim2=2)


def _unit_scale(K: torch.Tensor) -> torch.Tensor:
    """Divide pela diagonal media. Nao muda o resultado (a escala sai no T_tr); evita underflow."""
    return K / _diag(K).mean(1).clamp_min(1e-300)[:, None, None]


def _angular(K: torch.Tensor, shift: float) -> torch.Tensor:
    d = _diag(K).clamp_min(1e-12).sqrt()
    return (K / (d[:, :, None] * d[:, None, :]) + shift) / (1.0 + shift)


def joint_transform(K: torch.Tensor, mode: str = "trace_center") -> tuple[torch.Tensor, torch.Tensor]:
    """T_tr o T_c (ou so T_tr) em lote; devolve tambem as imagens cujo traco se anulou."""
    before = _diag(K).sum(1).abs()
    if mode == "trace_center":
        K = K - K.mean(1, keepdim=True) - K.mean(2, keepdim=True) + K.mean((1, 2), keepdim=True)
    elif mode != "trace":
        raise ValueError(f"transformacao desconhecida: {mode}")
    tr = _diag(K).sum(1)
    # 1e-6 fica acima da precisao do float32 em que as fontes sao guardadas: em b6 o K_Γ
    # constante deixa, depois do centramento, um residuo de ~1e-8 que e so arredondamento
    # e que, normalizado pelo proprio traco, viraria um "campo" de ruido nao PSD.
    degenerate = tr <= 1e-6 * before.clamp_min(1e-300)
    K = K / torch.where(degenerate, torch.ones_like(tr), tr)[:, None, None]
    # a matriz degenerada vira zero exato: sem isso o residuo de arredondamento seguiria
    # para o eigh e falharia a identidade G_ss = Σ λ V² (diagonal de ~1e-16)
    K = torch.where(degenerate[:, None, None], torch.zeros_like(K), K)
    return 0.5 * (K + K.transpose(1, 2)), degenerate


def pixel_grams(
    acts: torch.Tensor,
    grads: torch.Tensor,
    metric: torch.Tensor,
    device: torch.device,
    reps: tuple[str, ...] = REPS,
    component: str = "id",
    shift: float = 0.0,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Grams brutas (antes de T) de cada fonte e da conjunta, e o posto efetivo de K_Γ."""
    A = acts.to(device, torch.float64).flatten(2).transpose(1, 2)  # (B, P, C)
    G = grads.to(device, torch.float64).flatten(2).transpose(1, 2)
    M = metric.to(device, torch.float64)
    comps = {
        "activations": A @ A.transpose(1, 2),
        "grads": G @ G.transpose(1, 2),
        "params": A @ M @ A.transpose(1, 2),
    }
    k_gamma = comps["grads"]
    gamma_pr = _diag(k_gamma).sum(1) ** 2 / (k_gamma**2).sum((1, 2)).clamp_min(1e-300)
    for key, K in comps.items():
        comps[key] = _unit_scale(K) if component == "id" else _angular(K, shift)
    out = {}
    for rep in reps:
        if rep == "joint":
            out["joint"] = comps["params"] * comps["activations"] * comps["grads"]
        else:
            out[rep] = comps[rep]
    return out, gamma_pr


def pixel_energy(G: torch.Tensor, rule: str = "participation", tau: float = 0.95) -> dict:
    """Ê_sk = λ̂_k V_sk², E_∥, E_⊥, E_ratio e r por imagem, em lote."""
    vals, vecs = torch.linalg.eigh(G)
    vals, vecs = vals.flip(-1).clamp_min(0.0), vecs.flip(-1)
    mode = vals[:, None, :] * vecs**2  # (B, P, P)
    total = mode.sum(2)
    diag = _diag(G)
    scale = diag.abs().amax(1, keepdim=True).clamp_min(1e-300)
    # a identidade G_ss = Σ_k Ê_sk e o teste de sanidade da afericao, como em spectral.py
    if not torch.allclose(total / scale, diag / scale, atol=1e-6):
        raise AssertionError("G_ss != sum_k lambda_k V_sk^2: decomposicao inconsistente")
    s = vals.sum(1)
    if rule == "participation":
        r = torch.round(s**2 / (vals**2).sum(1).clamp_min(1e-300)).clamp_min(1).long()
    elif rule == "tau":
        r = (vals.cumsum(1) / s.clamp_min(1e-300)[:, None] < tau).sum(1) + 1
    else:
        raise ValueError(f"regra desconhecida: {rule}")
    r = r.clamp(max=G.shape[1])
    keep = torch.arange(G.shape[1], device=G.device)[None, :] < r[:, None]
    par = (mode * keep[:, None, :]).sum(2)
    return {
        "E_par": par,
        "E_perp": total - par,
        "E_total": total,
        "E_ratio": (total - par) / total.clamp_min(1e-300),
        "rank": r,
        "retained": (vals * keep).sum(1) / s.clamp_min(1e-300),
    }


def gradcam(acts: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
    """Grad-CAM no mesmo estado e com o mesmo gradiente: ReLU(Σ_c mean_s(g_c) a_c)."""
    alpha = grads.mean((2, 3), keepdim=True)
    return F.relu((alpha * acts).sum(1))


def compute_maps(
    src: PixelSources,
    device: torch.device,
    reps: tuple[str, ...] = REPS,
    component: str = "id",
    shift: float = 0.0,
    joint: str = "trace_center",
    rule: str = "participation",
    tau: float = 0.95,
    gpu_mb: float = 1500.0,
    max_side: int | None = None,
) -> dict:
    """Mapa E_∥ por representacao e bloco, mais Grad-CAM e os descritivos do espectro.

    `max_side` limita a grade de posicoes por pooling medio do estado e do gradiente
    (o eigh de uma Gram 4096x4096 custa ~2,4 s por imagem, contra 0,08 s em 1024x1024).
    O gradiente medio sobre uma janela e o gradiente com respeito ao estado medio dela
    so a menos de um fator constante, que sai no T_tr. `None` mantem a resolucao nativa.
    """
    maps = {rep: {} for rep in reps}
    stats = {rep: {} for rep in reps}
    e_ratio, cams, gamma_participation = {}, {}, {}
    for b in src.blocks:
        acts, grads = src.acts[b], src.grads[b]
        if max_side and acts.shape[-1] > max_side:
            acts = F.adaptive_avg_pool2d(acts, max_side)
            grads = F.adaptive_avg_pool2d(grads, max_side)
        n, _, h, w = acts.shape
        p = h * w
        chunk = max(1, int(gpu_mb * 2**20 // (p * p * 8 * 12)))
        par = {rep: [] for rep in reps}
        ranks = {rep: [] for rep in reps}
        retained = {rep: [] for rep in reps}
        degen = {rep: [] for rep in reps}
        ratio, gpr = [], []
        for s in range(0, n, chunk):
            grams, g_pr = pixel_grams(
                acts[s : s + chunk], grads[s : s + chunk], src.metric[b], device, reps, component, shift
            )
            gpr.append(g_pr.cpu())
            for rep, K in grams.items():
                G, deg = joint_transform(K, joint)
                e = pixel_energy(G, rule, tau)
                field = e["E_par"].clone()
                field[deg] = float("nan")
                par[rep].append(field.reshape(-1, h, w).cpu())
                ranks[rep].append(e["rank"].cpu())
                retained[rep].append(e["retained"].cpu())
                degen[rep].append(deg.cpu())
                if rep == "joint":
                    ratio.append(e["E_ratio"].reshape(-1, h, w).cpu())
            del grams
            if device.type == "cuda":
                torch.cuda.empty_cache()
        for rep in reps:
            m = torch.cat(par[rep]).numpy()
            maps[rep][b] = m
            flat = m.reshape(n, -1)
            with np.errstate(invalid="ignore", divide="ignore"):
                cv = flat.std(1) / np.abs(flat.mean(1))
            stats[rep][b] = {
                "rank_mean": float(torch.cat(ranks[rep]).float().mean()),
                "retained_mean": float(torch.cat(retained[rep]).mean()),
                "field_cv_mean": float(np.nanmean(cv)) if np.isfinite(cv).any() else float("nan"),
                "degenerate_fraction": float(torch.cat(degen[rep]).float().mean()),
            }
        if ratio:
            e_ratio[b] = torch.cat(ratio).numpy()
        cams[b] = gradcam(acts, grads).numpy()
        gamma_participation[b] = float(torch.cat(gpr).mean())
    return {
        "maps": maps,
        "E_ratio": e_ratio,
        "gradcam": cams,
        "stats": stats,
        "gamma_participation": gamma_participation,
    }


# --------------------------------------------------------------------------
# testes contra o comportamento da rede
# --------------------------------------------------------------------------


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    return float(((y[1:] + y[:-1]) * 0.5 * np.diff(x)).sum())


@torch.no_grad()
def deletion_insertion(
    model: torch.nn.Module,
    x: torch.Tensor,
    maps: np.ndarray,
    target: np.ndarray,
    device: torch.device,
    steps: int = 20,
    group: int = 24,
    seed: int = 0,
) -> dict:
    """Remove (ou insere) pixels na ordem decrescente do mapa e mede a margem da rede.

    O mapa, na resolucao do bloco, e levado a 32x32 por interpolacao bilinear; empates
    (inevitaveis em 8x8 ampliado) sao quebrados por um ruido minusculo com semente fixa.
    O pixel removido vai para 0 no espaco normalizado, isto e, a cor media do CIFAR-10.
    A margem e a do alvo limpo, dividida pela margem limpa: 1 = intacta, 0 = anulada.

    Por que a ordem aleatoria nao e piso aqui: remover pixels espalhados produz ruido de
    alta frequencia, que derruba a margem mais rapido do que remover qualquer regiao
    contigua -- medido, *todo* mapa (inclusive o da rede aleatoria) fica acima da ordem
    aleatoria na delecao. Por isso a leitura principal e o **gap LeRF - MoRF**: a mesma
    ordem aplicada do menos relevante para o mais relevante, que produz o mesmo tipo de
    padrao de delecao. O gap e ~0 para uma ordem aleatoria por construcao, e positivo
    quando o mapa separa os pixels que a rede usa dos que ela ignora.
    """
    n = len(x)
    size = x.shape[-2:]
    # `group` e calibrado para 32x32; cada lote passa group * 3 * (steps + 1) imagens pela rede,
    # entao o grupo encolhe com a area (6 em 64x64) para caber na memoria da GPU
    group = max(1, int(group * 1024 // (size[0] * size[1])))
    m = torch.from_numpy(np.nan_to_num(np.asarray(maps, dtype=np.float32), nan=0.0))[:, None]
    up = F.interpolate(m, size=size, mode="bilinear", align_corners=False)[:, 0].flatten(1)
    gen = torch.Generator().manual_seed(seed)
    span = up.abs().amax(1, keepdim=True).clamp_min(1e-30)
    order = torch.argsort(up + torch.rand(up.shape, generator=gen) * 1e-6 * span, dim=1, descending=True)
    npix = order.shape[1]
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(npix).expand_as(order).contiguous())
    counts = torch.round(torch.linspace(0, 1, steps + 1) * npix).long()
    fractions = (counts.float() / npix).numpy()
    t = torch.as_tensor(target, device=device)

    clean = contrastive_margin(model(x.to(device)), t)
    names = ("deletion", "insertion", "deletion_lerf")
    curves = {k: np.zeros((n, steps + 1)) for k in (*names, *(f"{k}_keep" for k in names))}
    for s in range(0, n, group):
        idx = torch.arange(s, min(s + group, n))
        g = len(idx)
        shape = (g, steps + 1, 1, *size)
        mask = (rank[idx][:, None, :] < counts[None, :, None]).reshape(shape).to(device)
        mask_lerf = ((npix - 1 - rank[idx])[:, None, :] < counts[None, :, None]).reshape(shape).to(device)
        xi = x[idx].to(device)[:, None]
        batch = torch.cat([
            (xi * ~mask).flatten(0, 1), (xi * mask).flatten(0, 1), (xi * ~mask_lerf).flatten(0, 1)
        ])
        logits = model(batch)
        tt = t[idx].repeat_interleave(steps + 1).repeat(len(names))
        marg = contrastive_margin(logits, tt) / clean[idx].repeat_interleave(steps + 1).repeat(len(names))
        keep = (logits.argmax(1) == tt).float()
        part = g * (steps + 1)
        for j, name in enumerate(names):
            curves[name][s : s + g] = marg[j * part : (j + 1) * part].reshape(g, -1).cpu().numpy()
            curves[f"{name}_keep"][s : s + g] = keep[j * part : (j + 1) * part].reshape(g, -1).cpu().numpy()

    auc = {k: _trapezoid(v.mean(0), fractions) for k, v in curves.items()}
    auc["gap"] = auc["deletion_lerf"] - auc["deletion"]
    auc["gap_keep"] = auc["deletion_lerf_keep"] - auc["deletion_keep"]
    return {
        "fractions": fractions,
        "mean_curves": {k: v.mean(0) for k, v in curves.items()},
        "auc": auc,
    }


def map_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman medio, imagem a imagem, entre dois mapas da mesma resolucao."""
    values = []
    for u, v in zip(a.reshape(len(a), -1), b.reshape(len(b), -1)):
        if not (np.isfinite(u).all() and np.isfinite(v).all()) or u.std() == 0 or v.std() == 0:
            continue
        ru, rv = rankdata(u), rankdata(v)
        values.append(float(np.corrcoef(ru, rv)[0, 1]))
    return float(np.mean(values)) if values else float("nan")

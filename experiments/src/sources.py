"""As tres fontes indexadas por objeto e a Gram conjunta de cada bloco.

Para um bloco com filtros de saida $W \\in \\mathbb{R}^{C \\times F}$ e estado pos-ReLU
$A_i \\in \\mathbb{R}^{C \\times P}$ do objeto $i$ (P = 1 num MLP, a grade espacial numa
ResNet), as fontes sao

  A  o estado, A_i
  Γ  a derivada da margem contrastiva com respeito ao estado, no mesmo formato
  W  o filtro efetivo que o objeto ativa, p_i[:, s] = sum_c A_i[c, s] W[c, :]

A Gram dos filtros de uma camada e a mesma para todo objeto e nao pode indexa-los; o filtro
efetivo resolve isso sem abandonar os parametros, e a Gram dos filtros reaparece como a forma
bilinear da fonte: <p_i, p_j> = u_i^T (W W^T) u_j. Como o peso de cada filtro e a propria
ativacao, W nao e independente de A -- medido, as distancias de W e de A tem Spearman
0,88–0,97 --, e isso vai reportado em vez de suposto pequeno.

As tres fontes tem o formato (N, D, P) e passam pelo mesmo `component_gram` (linear + angular
+ deslocamento, a unica composicao que realiza o produto tensorial de verdade).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .geometry import component_gram
from .kernels import apply_transform, hadamard, project_psd

SOURCE_KEYS = ("params", "activations", "grads")


@dataclass
class Extraction:
    """Fontes por bloco e por imagem, mais o comportamento da rede."""

    blocks: list[str]
    sources: dict[str, dict[str, np.ndarray]]  # bloco -> fonte -> (N, D, P)
    labels: np.ndarray
    preds: np.ndarray
    logits: np.ndarray
    trained: bool = True
    widths: dict[str, int] = field(default_factory=dict)

    @property
    def correct(self) -> np.ndarray:
        return self.preds == self.labels

    @property
    def logit_margin(self) -> np.ndarray:
        """Logit vencedor menos o vice: a confianca da rede naquela imagem."""
        top2 = np.sort(self.logits, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]

    @property
    def prob_max(self) -> np.ndarray:
        e = np.exp(self.logits - self.logits.max(1, keepdims=True))
        return (e / e.sum(1, keepdims=True)).max(1)

    def confusion(self, n_classes: int = 10) -> np.ndarray:
        m = np.zeros((n_classes, n_classes))
        for t, p in zip(self.labels, self.preds):
            m[t, p] += 1
        return m


def _param_source(act: torch.Tensor, weight: torch.Tensor) -> np.ndarray:
    """(N, C, P) x (C, F) -> (N, F, P): o filtro efetivo ativado por cada imagem."""
    p = torch.einsum("ncp,cf->nfp", act, weight)
    return p.cpu().numpy().astype(np.float32)


def extract_sources(
    model: torch.nn.Module,
    loader: DataLoader,
    blocks: list[str],
    device: torch.device,
    grid: int = 4,
    target_mode: str = "pred",
    fixed_target: int = 0,
    trained: bool = True,
) -> Extraction:
    """Um forward e um backward por lote, com hook na saida de cada bloco.

    O ponto de sondagem e a saida do bloco -- o estado residual pos-ReLU que o
    bloco seguinte le -- de modo que ha um tensor por bloco e nenhuma escolha
    sobre onde dentro dele o estado mora.

    `target_mode` decide o alvo da margem: `pred` (a classe predita, definicao da
    quantidade), `true` (rotulo verdadeiro) ou `fixed` (uma classe para todas as
    imagens, controle que remove o condicionamento por imagem).
    """
    store: dict[str, torch.Tensor] = {}
    handles = []

    def hook(name):
        def fn(_mod, _inputs, output):
            output.retain_grad()
            store[name] = output

        return fn

    for name in blocks:
        handles.append(model.get_block(name).register_forward_hook(hook(name)))

    weights = {
        name: model.get_block(name).out_conv.weight.detach().reshape(
            model.get_block(name).out_conv.out_channels, -1
        )
        for name in blocks
    }

    model.eval()
    chunks: dict[str, dict[str, list[np.ndarray]]] = {
        name: {k: [] for k in SOURCE_KEYS} for name in blocks
    }
    all_labels, all_preds, all_logits = [], [], []

    try:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            store.clear()
            logits = model(x)
            if target_mode == "fixed":
                target = torch.full((len(logits),), fixed_target, device=device, dtype=torch.long)
            elif target_mode == "true":
                target = y.to(device)
            else:
                target = logits.argmax(1)
            n, k = logits.shape
            rows = torch.arange(n, device=device)
            others = (logits.sum(1) - logits[rows, target]) / (k - 1)
            margin = logits[rows, target] - others

            model.zero_grad(set_to_none=True)
            margin.sum().backward()

            for name in blocks:
                state = store[name]
                act = F.adaptive_avg_pool2d(state.detach(), (grid, grid)).flatten(2)
                grad = F.adaptive_avg_pool2d(state.grad, (grid, grid)).flatten(2)
                chunks[name]["activations"].append(act.cpu().numpy().astype(np.float32))
                chunks[name]["grads"].append(grad.cpu().numpy().astype(np.float32))
                chunks[name]["params"].append(_param_source(act, weights[name]))

            all_labels.append(y.numpy())
            all_preds.append(logits.argmax(1).detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)

    return Extraction(
        blocks=list(blocks),
        sources={
            name: {k: np.concatenate(v) for k, v in per_source.items()}
            for name, per_source in chunks.items()
        },
        labels=np.concatenate(all_labels),
        preds=np.concatenate(all_preds),
        logits=np.concatenate(all_logits),
        trained=trained,
        widths={name: int(weights[name].shape[0]) for name in blocks},
    )


# --------------------------------------------------------------------------
# Grams das tres fontes e composicao tensorial
# --------------------------------------------------------------------------


def block_grams(
    block_sources: dict[str, np.ndarray],
    shift: float = 1.0,
    joint_transform: str = "trace_center",
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Gram de cada fonte e a conjunta das tres, transformadas, mais as Grams brutas.

    Cada componente e a Gram linear angular deslocada para [0, 1] (`component_gram`), e a
    composicao e o produto de Hadamard $G = G_W \\circ G_A \\circ G_\\Gamma$ da eq.
    `transformed_tensor_gram`. A transformacao final (T_tr o T_c por padrao) e aplicada a
    conjunta e a cada componente, para que todas sejam comparaveis. As Grams brutas (diagonal
    1, entradas em [0, 1]) vao em `_raw`: sao elas que dizem se uma representacao e degenerada.
    """
    components = {k: component_gram(block_sources[k], shift, device) for k in SOURCE_KEYS}
    out = {k: apply_transform(project_psd(v), joint_transform) for k, v in components.items()}
    raw_joint = hadamard([components[k] for k in SOURCE_KEYS])
    out["joint"] = apply_transform(project_psd(raw_joint), joint_transform)
    out["_raw"] = {**components, "joint": raw_joint}
    return out



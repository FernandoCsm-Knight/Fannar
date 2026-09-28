"""Espaco de representacao em 10 bases tabulares, com varias sementes.

Mede, por bloco do MLP residual, as leituras de `src/suite.py` -- geometria contra a decisao,
espectro, comparacao com A, Γ e RSA isolados, decomposicao contra a CKA e datacao por
intervencao -- para reportar media e desvio padrao e em que fracao das execucoes cada
afirmacao se sustenta.

Duas variantes do alvo da margem sao rodadas em cada execucao, sobre o mesmo modelo treinado:
o alvo predito (definicao do metodo) e o alvo fixo (classe 0 para todas as amostras). A
segunda e o controle necessario: no MLP o gradiente da margem da classe predita carrega a
propria predicao -- a classe predita explica 36% da variacao de Γ no `stem` e 100% no `b6` --
e as leituras referidas a predicao saturam em 1,000 por tautologia.

O que depende de posicao espacial (campo de energia, conjuntos de nivel) nao existe em dados
tabulares; para a pergunta "por que esta amostra teve outra classificacao", ver
`tabular_differences.py`.

Uso:
    python tabular_suite.py --seeds 5
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src import plots_suite as P
from src import suite as S
from src.data_tabular import DATASETS, balanced_sample, load_tabular, split, standardize
from src.mlp import ResidualMLP
from src.sources import _param_source


def train(model, X_tr, y_tr, X_va, y_va, device, args, seed) -> tuple[float, int]:
    """Adam com parada antecipada pela acuracia de validacao; devolve a melhor e a epoca."""
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    Xt, yt = torch.from_numpy(X_tr).to(device), torch.from_numpy(y_tr).to(device)
    Xv, yv = torch.from_numpy(X_va).to(device), torch.from_numpy(y_va).to(device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, best_epoch, wait = -1.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(Xt), generator=gen).to(device)
        for s in range(0, len(order), args.batch):
            idx = order[s : s + args.batch]
            if len(idx) < 2:  # BatchNorm precisa de mais de uma amostra
                continue
            opt.zero_grad(set_to_none=True)
            crit(model(Xt[idx]), yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(Xv).argmax(1) == yv).float().mean())
        if acc > best:
            best, best_state, best_epoch, wait = acc, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state)
    return best, best_epoch


def extract(model, X: np.ndarray, device: torch.device, target="pred", batch: int = 512) -> dict:
    """A, Γ e Wᵀh em cada bloco, com a margem contrastiva da classe `target` (ou da predita)."""
    blocks = model.block_names
    store: dict[str, torch.Tensor] = {}
    handles = []
    for b in blocks:
        def hook(_m, _i, out, b=b):
            out.retain_grad()
            store[b] = out
        handles.append(model.get_block(b).register_forward_hook(hook))
    weights = {b: model.get_block(b).out_conv.weight.detach() for b in blocks}
    acts, grads = {b: [] for b in blocks}, {b: [] for b in blocks}
    logits_all = []
    model.eval()
    try:
        for s in range(0, len(X), batch):
            store.clear()
            logits = model(torch.from_numpy(X[s : s + batch]).to(device))
            cls = (logits.argmax(1) if target == "pred"
                   else torch.full((len(logits),), int(target), device=device, dtype=torch.long))
            rows = torch.arange(len(logits), device=device)
            k = logits.shape[1]
            margin = logits[rows, cls] - (logits.sum(1) - logits[rows, cls]) / (k - 1)
            model.zero_grad(set_to_none=True)
            margin.sum().backward()
            for b in blocks:
                acts[b].append(store[b].detach())
                grads[b].append(store[b].grad.detach())
            logits_all.append(logits.detach().cpu().numpy())
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)
    sources = {}
    for b in blocks:
        a = torch.cat(acts[b])[:, :, None]  # (N, C, 1): a posicao espacial e um ponto so
        g = torch.cat(grads[b])[:, :, None]
        sources[b] = {"activations": a.cpu().numpy().astype(np.float32),
                      "grads": g.cpu().numpy().astype(np.float32),
                      "params": _param_source(a, weights[b])}
    logits = np.concatenate(logits_all)
    return {"sources": sources, "logits": logits, "preds": logits.argmax(1)}


def run_one(name: str, seed: int, args, device: torch.device) -> dict:
    X, y = load_tabular(name)
    train_idx, val_idx, test_idx = split(y, seed)
    X_tr, X_va, X_te = standardize(X[train_idx], X[val_idx], X[test_idx])
    y_tr, y_va, y_te = y[train_idx], y[val_idx], y[test_idx]
    n_classes = int(y.max() + 1)

    torch.manual_seed(seed)
    model = ResidualMLP(X.shape[1], n_classes, args.width, args.blocks).to(device)
    untrained = copy.deepcopy(model)  # os mesmos pesos iniciais: o piso so difere pelo treino
    val_acc, epoch = train(model, X_tr, y_tr, X_va, y_va, device, args, seed)
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(X_te).to(device)), 1).cpu().numpy()
    refs = S.behaviour_references(probs, y_te, n_classes)

    sample = balanced_sample(y_te, args.sample, seed)
    Xs, ys = X_te[sample], y_te[sample]
    blocks = model.block_names
    flips = S.flip_rates(model, torch.from_numpy(Xs), device, args.sigmas, seed)
    flips_u = S.flip_rates(untrained, torch.from_numpy(Xs), device, [2.0], seed)
    stability = [1.0 - flips[f"{b}@2"] for b in blocks]

    result = {"dataset": name, "seed": seed, "n_classes": n_classes, "val_acc": val_acc,
              "test_acc": float((probs.argmax(1) == y_te).mean()), "epochs": epoch,
              "n_sample": len(ys), "blocks": blocks, "flip": flips, "flip_untrained": flips_u,
              "test_zero_pairs": refs["test_zero_pairs"], "variants": {}}
    for variant, target in (("pred", "pred"), ("fixed", args.fixed_target)):
        ext, ext_u = extract(model, Xs, device, target), extract(untrained, Xs, device, target)
        result["variants"][variant] = S.analyse(ext["sources"], ext_u["sources"], blocks, ys,
                                                ext["preds"], ext["logits"], refs, stability,
                                                args, device)
        result["sample_errors"] = int((ext["preds"] != ys).sum())
        result["sample_zero_pairs"] = S.sample_zero_pairs(ys, ext["preds"], n_classes)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--sample", type=int, default=500, help="objetos da Gram, tirados do teste")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--permutations", type=int, default=20)
    ap.add_argument("--min-errors", type=int, default=10)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    ap.add_argument("--fixed-target", type=int, default=0, help="classe do controle de alvo fixo")
    ap.add_argument("--out", default="outputs/tabular_suite")
    args = ap.parse_args()

    out = Path(args.out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keys = [(name, seed) for name in args.datasets for seed in range(args.seeds)]
    runs = S.run_all(keys, lambda name, seed: run_one(name, seed, args, device), out)
    S.report(runs, out, list(DATASETS), "Espaço de representação em bases tabulares", P.plot_suite)
    print(f"pronto: {out}/")


if __name__ == "__main__":
    main()

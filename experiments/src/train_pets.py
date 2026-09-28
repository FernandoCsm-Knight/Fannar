"""Treina a ResNet nas 37 racas do Oxford-IIIT Pet, 128x128, uma semente por chamada.

Mesma arquitetura de sondagem (stem + 6 blocos residuais de larguras 64/128/256 + pooling
global) com passo 2 na convolucao de entrada: os estados ficam em 64x64 (stem–b2), 32x32
(b3–b4) e 16x16 (b5–b6). Treino do zero, sem pesos pre-treinados.

Da particao oficial `trainval`, 10% (estratificados) ficam para escolher o melhor ponto de
controle; o `test` oficial nao e tocado aqui. O checkpoint guarda a particao, as estatisticas
de normalizacao e o estado inicial, que e o controle nao treinado com os mesmos pesos.

    python -m src.train_pets --seed 0
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from .data_pets import SIZE, PetsDataset, channel_stats, load_pets, make_loader
from .model import CIFAR10ResNet
from .vit import PetsViT

ARCHS = ("resnet", "vit")


def make_model(n_classes: int, arch: str = "resnet") -> torch.nn.Module:
    """A ResNet do artigo ou o transformer de `src/vit.py`, mesma interface de sondagem."""
    if arch == "vit":
        return PetsViT(n_classes=n_classes, size=SIZE)
    return CIFAR10ResNet(n_classes=n_classes, stem_stride=2)


def optimizer_for(model, arch: str, epochs: int, steps: int):
    """SGD/OneCycle para a ResNet; AdamW/OneCycle para o transformer (nao treina com SGD)."""
    if arch == "vit":
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
        return opt, torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, epochs=epochs,
                                                        steps_per_epoch=steps, pct_start=0.15)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
    return opt, torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.05, epochs=epochs, steps_per_epoch=steps)


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha_mix: float = 0.2, alpha_cut: float = 1.0):
    """Mixup ou CutMix (metade a metade); devolve (x, y_a, y_b, lambda).

    Com 90 imagens por raca, misturar pares e o aumento que mais rende: cada passo ve uma
    combinacao nova em vez de repetir as mesmas fotos. A validacao nunca e misturada.
    """
    perm = torch.randperm(len(x), device=x.device)
    if torch.rand(1).item() < 0.5:
        lam = float(np.random.beta(alpha_mix, alpha_mix))
        return lam * x + (1 - lam) * x[perm], y, y[perm], lam
    lam = float(np.random.beta(alpha_cut, alpha_cut))
    h, w = x.shape[-2:]
    rh, rw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
    r, c = np.random.randint(0, max(1, h - rh + 1)), np.random.randint(0, max(1, w - rw + 1))
    x = x.clone()
    x[:, :, r : r + rh, c : c + rw] = x[perm][:, :, r : r + rh, c : c + rw]
    return x, y, y[perm], 1.0 - (rh * rw) / (h * w)


def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
            total += len(y)
    return correct / total


def train_seed(seed: int, out: Path, epochs: int = 60, batch: int = 64, workers: int = 8,
               device: torch.device | None = None, arch: str = "resnet", mix: float = 0.5) -> dict:
    """Treina uma semente e grava o checkpoint (com o estado inicial); devolve o blob."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_pets(SIZE)
    trainval = np.flatnonzero(data["split"] == 0)
    train_idx, val_idx = train_test_split(trainval, test_size=0.1, stratify=data["labels"][trainval],
                                          random_state=seed)
    mean, std = channel_stats(data["images"][train_idx])
    n_classes = len(data["classes"])

    torch.manual_seed(seed)
    model = make_model(n_classes, arch).to(device)
    initial = copy.deepcopy(model.state_dict())
    train_dl = make_loader(PetsDataset(data["images"][train_idx], data["labels"][train_idx], mean, std, True),
                           batch, shuffle=True, drop_last=True, workers=workers)
    val_dl = make_loader(PetsDataset(data["images"][val_idx], data["labels"][val_idx], mean, std),
                         256, workers=workers)
    opt, sched = optimizer_for(model, arch, epochs, len(train_dl))
    crit = nn.CrossEntropyLoss(label_smoothing=0.05 if arch == "resnet" else 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, best_state, best_epoch, start = -1.0, None, 0, 1
    # retomada: o estado completo e salvo a cada epoca, e uma interrupcao (queda de energia,
    # desligamento) continua da ultima epoca concluida em vez de recomecar
    partial = out.with_suffix(".partial.pt")
    if partial.exists():
        state = torch.load(partial, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        initial, best, best_state, best_epoch = state["initial"], state["best"], state["best_state"], state["best_epoch"]
        start = state["epoch"] + 1
        torch.set_rng_state(state["rng"])
        print(f"semente {seed}: retomando da época {start}", flush=True)
    for epoch in range(start, epochs + 1):
        model.train()
        t0 = time.time()
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            mixed = mix > 0 and torch.rand(1).item() < mix
            if mixed:
                x, ya, yb, lam = mix_batch(x, y)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
                loss = ((lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)) if mixed
                        else crit(logits, y))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        acc = evaluate(model, val_dl, device)
        if acc > best:
            best, best_state, best_epoch = acc, copy.deepcopy(model.state_dict()), epoch
        print(f"semente {seed} época {epoch:2d}/{epochs}: validação {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = partial.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "initial": initial, "best": best,
                    "best_state": best_state, "best_epoch": best_epoch, "epoch": epoch,
                    "rng": torch.get_rng_state()}, tmp)
        tmp.replace(partial)
    blob = {"state_dict": best_state, "initial_state_dict": initial, "val_acc": best, "epoch": best_epoch,
            "mean": mean, "std": std, "train_idx": train_idx, "val_idx": val_idx, "seed": seed,
            "n_classes": n_classes, "size": SIZE, "arch": arch}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    torch.save(blob, tmp)
    tmp.replace(out)
    partial.unlink(missing_ok=True)
    return blob


def checkpoint_path(seed: int, arch: str = "resnet", out_dir: Path = Path("outputs/pets_models")) -> Path:
    return out_dir / (f"seed{seed}.pt" if arch == "resnet" else f"{arch}_seed{seed}.pt")


def load_or_train(seed: int, out_dir: Path = Path("outputs/pets_models"), arch: str = "resnet", **kw) -> dict:
    path = checkpoint_path(seed, arch, out_dir)
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    return train_seed(seed, path, arch=arch, **kw)


def models_from(blob: dict, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    """(treinada, nao treinada com os mesmos pesos iniciais), ambas em modo de avaliacao."""
    arch = blob.get("arch", "resnet")
    trained, untrained = make_model(blob["n_classes"], arch), make_model(blob["n_classes"], arch)
    trained.load_state_dict(blob["state_dict"])
    untrained.load_state_dict(blob["initial_state_dict"])
    return trained.to(device).eval(), untrained.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--arch", default="resnet", choices=ARCHS)
    ap.add_argument("--mix", type=float, default=0.5, help="probabilidade de mixup/cutmix por lote")
    ap.add_argument("--out", default="outputs/pets_models")
    args = ap.parse_args()
    blob = train_seed(args.seed, checkpoint_path(args.seed, args.arch, Path(args.out)),
                      epochs=args.epochs, arch=args.arch, mix=args.mix)
    print(f"melhor validação {blob['val_acc']:.3f} na época {blob['epoch']}")


if __name__ == "__main__":
    main()

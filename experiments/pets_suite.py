"""Espaco de representacao no Oxford-IIIT Pet (37 racas), com varias sementes.

As mesmas leituras de `tabular_suite.py`, com as mesmas funcoes (`src/suite.py`), sobre a
ResNet de `src/train_pets.py`: os objetos sao imagens, e as tres fontes de cada bloco vem de
`src/sources.py` (estado pos-ReLU, gradiente da margem e filtro efetivo ativado, reduzidos a
uma grade 4x4). O desvio padrao e entre sementes: cada semente e um treino diferente, com a
sua propria inicializacao, que e tambem o controle nao treinado daquela semente.

A comparacao das regioes (campo de energia, conjuntos de nivel, diferenca entre imagens) fica
em `pets_regions.py`; aqui tudo e por imagem.

Uso:
    python pets_suite.py --seeds 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src import plots_suite as P
from src import suite as S
from src.data_pets import PetsDataset, balanced_subset, load_pets, make_loader
from src.sources import extract_sources
from src.train_pets import load_or_train, models_from


@torch.no_grad()
def probabilities(model, loader, device) -> np.ndarray:
    return np.concatenate([torch.softmax(model(x.to(device)), 1).cpu().numpy() for x, _ in loader])


def run_one(name: str, seed: int, args, device: torch.device) -> dict:
    blob = load_or_train(seed, arch=args.arch, epochs=args.epochs)
    data = load_pets(blob["size"])
    model, untrained = models_from(blob, device)
    test = np.flatnonzero(data["split"] == 1)
    labels_te = data["labels"][test]
    n_classes = blob["n_classes"]
    test_dl = make_loader(PetsDataset(data["images"][test], labels_te, blob["mean"], blob["std"]),
                          128, workers=args.workers)
    probs = probabilities(model, test_dl, device)
    refs = S.behaviour_references(probs, labels_te, n_classes)

    sample = test[balanced_subset(labels_te, args.sample // n_classes, seed)]
    labels = data["labels"][sample]
    ds = PetsDataset(data["images"][sample], labels, blob["mean"], blob["std"])
    loader = make_loader(ds, args.batch, workers=args.workers)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    blocks = model.block_names
    flips = S.flip_rates(model, x, device, args.sigmas, seed)
    flips_u = S.flip_rates(untrained, x, device, [2.0], seed)
    stability = [1.0 - flips[f"{b}@2"] for b in blocks]

    result = {"dataset": name, "seed": seed, "n_classes": n_classes, "val_acc": blob["val_acc"],
              "test_acc": float((probs.argmax(1) == labels_te).mean()), "epochs": blob["epoch"],
              "n_sample": len(labels), "blocks": blocks, "flip": flips, "flip_untrained": flips_u,
              "test_zero_pairs": refs["test_zero_pairs"], "variants": {}}
    for variant, mode in (("pred", "pred"), ("fixed", "fixed")):
        ext = extract_sources(model, loader, blocks, device, grid=args.grid, target_mode=mode,
                              fixed_target=args.fixed_target)
        ext_u = extract_sources(untrained, loader, blocks, device, grid=args.grid, target_mode=mode,
                                fixed_target=args.fixed_target, trained=False)
        result["variants"][variant] = S.analyse(ext.sources, ext_u.sources, blocks, labels, ext.preds,
                                                ext.logits, refs, stability, args, device)
        result["sample_errors"] = int((ext.preds != labels).sum())
        result["sample_zero_pairs"] = S.sample_zero_pairs(labels, ext.preds, n_classes)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--workers", type=int, default=4, help="workers persistentes dos DataLoaders")
    ap.add_argument("--arch", default="resnet", choices=("resnet", "vit"),
                    help="a ResNet do artigo ou o transformer (src/vit.py)")
    ap.add_argument("--sample", type=int, default=500, help="objetos da Gram, tirados do teste")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--permutations", type=int, default=20)
    ap.add_argument("--min-errors", type=int, default=10)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    ap.add_argument("--fixed-target", type=int, default=0)
    ap.add_argument("--out", default="outputs/pets_suite")
    args = ap.parse_args()

    out = Path(args.out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keys = [("pets", seed) for seed in range(args.seeds)]
    runs = S.run_all(keys, lambda name, seed: run_one(name, seed, args, device), out)
    title = "Espaço de representação no Oxford-IIIT Pet"
    S.report(runs, out, ["pets"], title + (" — transformer" if args.arch == "vit" else ""), P.plot_suite)
    print(f"pronto: {out}/")


if __name__ == "__main__":
    main()

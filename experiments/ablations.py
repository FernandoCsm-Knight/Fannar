"""Ablacoes da composicao e referencias de confianca e de vizinhanca (itens a-c da Subsecao
`subsec:adicionais`).

Mesmo protocolo das suites (`tabular_suite.py`, `pets_suite.py`): mesmas bases, sementes,
amostra de 500 objetos do teste, alvo fixo e piso cruzado. Muda so o que se compara (ver
`src/ablation.py`):

  (a) composicao   produto de Hadamard × soma (= Gram da concatenacao) × produto sem T_a ×
                   produto sem J × sem os dois, e as fontes A e Ω isoladas;
  (b) confianca    AUC da margem geometrica para o acerto × softmax maximo, margem dos logits,
                   trust score e Mahalanobis;
  (c) vizinhanca   k-vizinhos nos logits e nas ativacoes do ultimo bloco.

Nas bases tabulares o MLP e re-treinado com a mesma semente e a mesma receita de
`tabular_suite.py` (o treino e deterministico na mesma maquina, mas nao entre GPUs; os valores
da conjunta podem diferir na terceira casa dos da suite). No Pet os modelos salvos em
`outputs/pets_models/` sao reaproveitados; sem eles, sao treinados como em `pets_suite.py`.

Uso:
    python ablations.py --family tabular --seeds 5
    python ablations.py --family pets --arch resnet --seeds 3
    python ablations.py --family pets --arch vit --seeds 3

Saidas em `outputs/ablations/<familia>/`: `runs/*.json` (retomavel), `summary.json`, `tables.md`.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from src import ablation as A
from src import suite as S


def run_tabular(name: str, seed: int, args, device: torch.device) -> dict:
    from src.data_tabular import balanced_sample, load_tabular, split, standardize
    from src.mlp import ResidualMLP
    from tabular_suite import extract, train

    X, y = load_tabular(name)
    train_idx, val_idx, test_idx = split(y, seed)
    X_tr, X_va, X_te = standardize(X[train_idx], X[val_idx], X[test_idx])
    y_tr, y_va, y_te = y[train_idx], y[val_idx], y[test_idx]
    n_classes = int(y.max() + 1)

    torch.manual_seed(seed)
    model = ResidualMLP(X.shape[1], n_classes, args.width, args.blocks).to(device)
    untrained = copy.deepcopy(model)
    train(model, X_tr, y_tr, X_va, y_va, device, args, seed)
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(X_te).to(device)), 1).cpu().numpy()
    refs = S.behaviour_references(probs, y_te, n_classes)

    sample = balanced_sample(y_te, args.sample, seed)
    Xs, ys = X_te[sample], y_te[sample]
    ext = extract(model, Xs, device, args.fixed_target)
    ext_u = extract(untrained, Xs, device, args.fixed_target)
    blocks = model.block_names
    ref = np.random.default_rng(seed).choice(len(X_tr), size=min(args.train_ref, len(X_tr)), replace=False)
    train_feats = extract(model, X_tr[ref], device, args.fixed_target)["sources"][blocks[-1]]["activations"]
    out = A.analyse(ext["sources"], ext_u["sources"], blocks, ys, ext["preds"], ext["logits"], refs,
                    _flat(ext["sources"][blocks[-1]]["activations"]), _flat(train_feats), y_tr[ref],
                    args, device)
    return {"dataset": name, "seed": seed, "n_classes": n_classes, "blocks": blocks,
            "test_acc": float((probs.argmax(1) == y_te).mean()),
            "sample_errors": int((ext["preds"] != ys).sum()), **out}


def run_pets(name: str, seed: int, args, device: torch.device) -> dict:
    from src.data_images import load_images
    from src.data_pets import PetsDataset, balanced_subset, make_loader
    from src.sources import extract_sources
    from src.train_pets import load_or_train, models_from

    blob = load_or_train(seed, arch=args.arch, epochs=args.epochs, dataset=name, workers=args.workers)
    data = load_images(name)
    model, untrained = models_from(blob, device)
    n_classes = blob["n_classes"]
    test = np.flatnonzero(data["split"] == 1)
    labels_te = data["labels"][test]

    def loader_of(idx):
        ds = PetsDataset(data["images"][idx], data["labels"][idx], blob["mean"], blob["std"])
        return make_loader(ds, args.batch, workers=args.workers)

    with torch.no_grad():
        probs = np.concatenate([torch.softmax(model(x.to(device)), 1).cpu().numpy()
                                for x, _ in loader_of(test)])
    refs = S.behaviour_references(probs, labels_te, n_classes)

    sample = test[balanced_subset(labels_te, args.sample // n_classes, seed)]
    labels = data["labels"][sample]
    blocks = model.block_names
    kw = dict(grid=args.grid, target_mode="fixed", fixed_target=args.fixed_target)
    ext = extract_sources(model, loader_of(sample), blocks, device, **kw)
    ext_u = extract_sources(untrained, loader_of(sample), blocks, device, trained=False, **kw)

    trainval = np.flatnonzero(data["split"] == 0)
    ref = trainval[balanced_subset(data["labels"][trainval], args.train_ref // n_classes, seed)]
    ext_ref = extract_sources(model, loader_of(ref), [blocks[-1]], device, **kw)
    # media sobre as posicoes: o vetor que a cabeca le, e o espaco usual do Mahalanobis
    feats = ext.sources[blocks[-1]]["activations"].mean(-1)
    train_feats = ext_ref.sources[blocks[-1]]["activations"].mean(-1)
    out = A.analyse(ext.sources, ext_u.sources, blocks, labels, ext.preds, ext.logits, refs,
                    feats.astype(np.float64), train_feats.astype(np.float64), data["labels"][ref],
                    args, device)
    return {"dataset": name, "seed": seed, "n_classes": n_classes, "blocks": blocks,
            "test_acc": float((probs.argmax(1) == labels_te).mean()),
            "sample_errors": int((ext.preds != labels).sum()), **out}


def _flat(a: np.ndarray) -> np.ndarray:
    return a.reshape(len(a), -1).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=("tabular", "pets", "cifar100"), default="tabular")
    ap.add_argument("--datasets", nargs="+", default=None, help="so no tabular; padrao: as dez bases")
    ap.add_argument("--seeds", type=int, default=None, help="padrao: 5 no tabular, 3 no Pet")
    ap.add_argument("--arch", default="resnet", choices=("resnet", "vit"))
    ap.add_argument("--sample", type=int, default=None, help="padrao: 500; 1000 no CIFAR-100")
    ap.add_argument("--train-ref", type=int, default=5000,
                    help="amostras do treino para o trust score e o Mahalanobis")
    # receita do MLP: a mesma de tabular_suite.py
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=None, help="padrao: 100 no tabular, 60 no Pet")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=None, help="padrao: 256 no tabular, 50 no Pet")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    # leituras: as mesmas das suites
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--permutations", type=int, default=20)
    ap.add_argument("--min-errors", type=int, default=10)
    ap.add_argument("--fixed-target", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tabular = args.family == "tabular"
    args.seeds = args.seeds or (5 if tabular else 3)
    args.epochs = args.epochs or (100 if tabular else 60)
    args.batch = args.batch or (256 if tabular else 50)
    args.sample = args.sample or (1000 if args.family == "cifar100" else 500)
    tag = "tabular" if tabular else f"{args.family}_{args.arch}"
    out = Path(args.out or f"outputs/ablations/{tag}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tabular:
        from src.data_tabular import DATASETS
        names = args.datasets or list(DATASETS)
        keys = [(n, s) for n in names for s in range(args.seeds)]
        fn = run_tabular
    else:
        keys = [(args.family, s) for s in range(args.seeds)]
        fn = run_pets

    (out / "runs").mkdir(parents=True, exist_ok=True)
    runs = []
    for name, seed in keys:
        path = out / "runs" / f"{name}_s{seed}.json"
        if path.exists():
            runs.append(json.loads(path.read_text()))
            continue
        r = fn(name, seed, args, device)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, ensure_ascii=False, default=float))
        tmp.replace(path)
        runs.append(r)
        mid = r["blocks"][len(r["blocks"]) // 2]
        m = r["metrics"][mid]
        print(f"{name:14s} semente {seed}: AUC do acerto em {mid} — produto {m['joint']['correctness_auc']:.3f} "
              f"soma {m['additive']['correctness_auc']:.3f} sem T_a {m['no_angular']['correctness_auc']:.3f} "
              f"sem J {m['no_shift']['correctness_auc']:.3f} | softmax {r['confidence']['softmax']:.3f}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = A.summarize(runs)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, default=float))
    A.write_tables(summary, out / "tables.md", f"Ablações da composição — {tag}")
    print(f"pronto: {out}/")


if __name__ == "__main__":
    main()

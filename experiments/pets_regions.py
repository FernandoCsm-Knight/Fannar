"""Campo de energia e conjuntos de nivel no Oxford-IIIT Pet, conferidos contra as mascaras.

Objetos: as posicoes espaciais de cada imagem (grade de ate 32x32 por bloco). Duas leituras:

  campo de energia   Ê_∥(s) = Σ_{k≤r} λ̂_k V_sk² na Gram conjunta das posicoes, com a
                     intensidade de cada posicao (`src/pixel_energy.py`);
  conjuntos de nivel a distancia angular a uma posicao tipica do animal (s0), com as faixas
                     de nivel e a vizinhanca N_k(s0) (`src/level_sets.py`).

O que a base acrescenta e a **mascara do animal** em cada imagem, que nunca entra no treino:
ela transforma "o campo cai no animal" e "a regiao equivalente a s0 e o animal" em numeros.
As referencias sao obrigatorias, porque os animais estao quase sempre no meio da foto:

  acaso        a fracao da imagem ocupada pelo animal (o que um mapa uniforme acerta)
  centro       um mapa gaussiano que so marca o centro da imagem
  piso         o mesmo campo na rede nao treinada, com os mesmos pesos iniciais
  Grad-CAM     a referencia de literatura, com o mesmo estado e o mesmo gradiente

e o teste de delecao (gap LeRF − MoRF da margem) diz se o campo aponta o que a rede usa.

Uso:
    python pets_regions.py --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src import plots_energy as P
from src.data_pets import PetsDataset, balanced_subset, load_pets
from src.level_sets import center_prior, image_gram, mask_to_grid, readout
from src.pixel_energy import REPS, compute_maps, deletion_insertion, extract_pixel_sources, map_spearman
from src.report import clean, fmt, table
from src.train_pets import load_or_train, models_from

TITLES = {"joint": r"conjunta $(W,A,\Gamma)$", "activations": r"só $A$", "params": r"só $W$",
          "grads": r"só $\Gamma$", "gradcam": "Grad-CAM", "cross": "conjunta, rede aleatória",
          "center": "só o centro", "random": "aleatório"}


def mask_stats(fields: np.ndarray, masks_grid: np.ndarray) -> dict:
    """Fracao da energia do campo dentro do animal, o acaso e o acerto do ponto maximo."""
    inside, chance, pointing = [], [], []
    for f, m in zip(fields, masks_grid):
        if not np.isfinite(f).all() or f.sum() <= 0:
            continue
        f = np.maximum(f, 0.0)
        inside.append(float((f * m).sum() / f.sum()))
        chance.append(float(m.mean()))
        pointing.append(float(m.reshape(-1)[f.reshape(-1).argmax()] >= 0.5))
    if not inside:
        return {"inside": np.nan, "chance": np.nan, "lift": np.nan, "pointing": np.nan}
    return {"inside": float(np.mean(inside)), "chance": float(np.mean(chance)),
            "lift": float(np.mean(inside) - np.mean(chance)), "pointing": float(np.mean(pointing))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="resnet", choices=("resnet", "vit"),
                    help="a ResNet do artigo ou o transformer (src/vit.py)")
    ap.add_argument("--per-class", type=int, default=3, help="imagens de teste por raça")
    ap.add_argument("--batch", type=int, default=37)
    ap.add_argument("--max-side", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--random-seeds", type=int, default=2)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--example-block", default="b4")
    ap.add_argument("--bands", type=int, default=6)
    ap.add_argument("--display-gamma", type=float, default=0.5)
    ap.add_argument("--gpu-mb", type=float, default=1500.0)
    ap.add_argument("--out", default=None, help="padrão: outputs/pets_regions[_vit]")
    args = ap.parse_args()
    P.DISPLAY_GAMMA = args.display_gamma

    out = Path(args.out or ("outputs/pets_regions" if args.arch == "resnet" else "outputs/pets_regions_vit"))
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = load_or_train(args.seed, arch=args.arch)
    data = load_pets(blob["size"])
    model, untrained = models_from(blob, device)
    test = np.flatnonzero(data["split"] == 1)
    idx = test[balanced_subset(data["labels"][test], args.per_class, args.seed)]
    labels, images, masks = data["labels"][idx], data["images"][idx], data["masks"][idx]
    ds = PetsDataset(images, labels, blob["mean"], blob["std"])
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    blocks = model.block_names
    classes = [str(c) for c in data["classes"]]
    print(f"{len(x)} imagens de teste, {len(classes)} raças, device={device}")

    opts = dict(component="id", shift=0.0, joint="trace_center", rule="participation", tau=0.95,
                gpu_mb=args.gpu_mb, max_side=args.max_side)
    src = extract_pixel_sources(model, x, labels, blocks, device, args.batch, "pred")
    preds = src.preds
    res = compute_maps(src, device, REPS, **opts)
    src_u = extract_pixel_sources(untrained, x, labels, blocks, device, args.batch, "pred")
    res_u = compute_maps(src_u, device, ("joint",), **opts)

    fields = {rep: res["maps"][rep] for rep in REPS} | {"gradcam": res["gradcam"],
                                                       "cross": res_u["maps"]["joint"]}
    rng = np.random.default_rng(args.seed)
    grids = {b: res["maps"]["joint"][b].shape[1:] for b in blocks}
    fields["center"] = {b: np.repeat(center_prior(*grids[b])[None], len(x), 0) for b in blocks}
    fields["random"] = {b: rng.random((len(x), *grids[b])) for b in blocks}
    masks_grid = {b: np.stack([mask_to_grid(m, *grids[b]) for m in masks]) for b in blocks}

    print("máscaras e deleção...")
    stats = {k: {b: mask_stats(v[b], masks_grid[b]) for b in blocks} for k, v in fields.items()}
    gap = {}
    for k, per_block in fields.items():
        gap[k] = {b: deletion_insertion(model, x, per_block[b], preds, device, args.steps,
                                        seed=args.seed)["auc"]["gap"]
                  if np.isfinite(per_block[b]).any() else float("nan") for b in blocks}

    # sanidade (Adebayo et al., 2018): um campo que nao muda com pesos aleatorios descreve a imagem,
    # nao a rede. Spearman entre o campo da rede treinada e o da aleatoria, imagem a imagem.
    sanity = {"joint": {b: map_spearman(res["maps"]["joint"][b], res_u["maps"]["joint"][b]) for b in blocks},
              "gradcam": {b: map_spearman(res["gradcam"][b], res_u["gradcam"][b]) for b in blocks}}

    print("conjuntos de nível...")
    level, level_u, examples = {}, {}, {}
    half = max(1, args.examples)
    keep = list(range(0, len(x), max(1, len(x) // half)))[:half]
    for b in blocks:
        rows, rows_u, p0_maps, p0_maps_u = [], [], [], []
        for i in range(len(x)):
            r = readout(*image_gram(src, i, b, device, args.max_side))
            ru = readout(*image_gram(src_u, i, b, device, args.max_side))
            m = masks_grid[b][i] >= 0.5
            for store, rr in ((rows, r), (rows_u, ru)):
                store.append({"knn_on_animal": float(m[rr["knn"]].mean()), "chance": float(m.mean()),
                              "s0_on_animal": float(m[rr["s0"]]), "s1_on_background": float(not m[rr["s1"]]),
                              "rho_energy": rr["rho_energy"], "rho_spatial": rr["rho_spatial"]})
            p0_maps.append(-r["p0"])
            p0_maps_u.append(-ru["p0"])
            if b == args.example_block and i in keep:
                examples[i] = r
        agg = lambda rs: {k: float(np.nanmean([r[k] for r in rs])) for k in rs[0]}
        level[b], level_u[b] = agg(rows), agg(rows_u)
        # delecao pela distancia a s0: a regiao equivalente a s0 sustenta a decisao?
        level[b]["gap"] = deletion_insertion(model, x, np.stack(p0_maps), preds, device, args.steps,
                                             seed=args.seed)["auc"]["gap"]
        level_u[b]["gap"] = deletion_insertion(model, x, np.stack(p0_maps_u), preds, device, args.steps,
                                               seed=args.seed)["auc"]["gap"]
        print(f"  {b}: N_k no animal {level[b]['knn_on_animal']:.3f} (acaso {level[b]['chance']:.3f}, "
              f"rede aleatória {level_u[b]['knn_on_animal']:.3f}), gap {level[b]['gap']:.3f}")

    accuracy = float((preds == labels).mean())
    report = {"config": vars(args), "n_images": len(x), "accuracy_sample": accuracy,
              "test_acc_model": blob.get("val_acc"), "blocks": blocks, "mask": stats,
              "deletion_gap": gap, "sanity": sanity, "level_sets": level, "level_sets_untrained": level_u}
    (out / "report.json").write_text(json.dumps(clean(report), indent=2, ensure_ascii=False))

    order = ["joint", "activations", "params", "grads", "gradcam", "cross", "center", "random"]
    md = [f"# Oxford-IIIT Pet — campo de energia e conjuntos de nível ({len(x)} imagens de teste, "
          f"acurácia na amostra {fmt(accuracy)})\n",
          table("Fração da energia do campo dentro do animal (acaso = fração da imagem ocupada pelo animal)",
                {TITLES[k]: {b: stats[k][b]["inside"] for b in blocks} for k in order}
                | {"acaso": {b: stats["joint"][b]["chance"] for b in blocks}}, blocks),
          table("Acerto do ponto máximo do campo (cai no animal?)",
                {TITLES[k]: {b: stats[k][b]["pointing"] for b in blocks} for k in order}, blocks),
          table("Gap LeRF − MoRF da margem (o campo aponta o que a rede usa?)",
                {TITLES[k]: gap[k] for k in order}, blocks),
          table("Sanidade: Spearman entre o campo da rede treinada e o da aleatória (baixo = depende dos pesos)",
                {TITLES[k]: sanity[k] for k in sanity}, blocks),
          table("Conjuntos de nível: fração de N_k(s0) sobre o animal",
                {"rede treinada": {b: level[b]["knn_on_animal"] for b in blocks},
                 "rede aleatória": {b: level_u[b]["knn_on_animal"] for b in blocks},
                 "acaso": {b: level[b]["chance"] for b in blocks}}, blocks),
          table("Conjuntos de nível: s0 no animal e s1 no fundo",
                {"s0 no animal": {b: level[b]["s0_on_animal"] for b in blocks},
                 "s1 no fundo": {b: level[b]["s1_on_background"] for b in blocks}}, blocks),
          table("Conjuntos de nível: gap de deleção pela distância a s0 (longe − perto)",
                {"rede treinada": {b: level[b]["gap"] for b in blocks},
                 "rede aleatória": {b: level_u[b]["gap"] for b in blocks}}, blocks),
          table("Conjuntos de nível: Spearman da distância com a energia e com a posição",
                {"com a energia": {b: level[b]["rho_energy"] for b in blocks},
                 "com a distância espacial": {b: level[b]["rho_spatial"] for b in blocks}}, blocks, 2)]
    (out / "tables.md").write_text("\n".join(md))

    print("figuras...")
    ex = sorted(examples)
    row_labels = [(f"{classes[labels[i]]}\n→ {classes[preds[i]]}" if preds[i] != labels[i]
                   else classes[labels[i]]).replace("_", " ") for i in ex]
    P.plot_field_grid(images[ex], {b: res["maps"]["joint"][b][ex] for b in blocks}, blocks, blocks,
                      row_labels, str(out / "energy_fields_blocks.png"), masks=masks[ex])
    cols = ["joint", "activations", "grads", "gradcam", "cross", "center"]
    eb = args.example_block
    P.plot_field_grid(images[ex], {k: fields[k][eb][ex] for k in cols}, cols, [TITLES[k] for k in cols],
                      row_labels, str(out / f"energy_fields_{eb}.png"), masks=masks[ex])
    P.plot_level_set_detail(images[ex], [examples[i] for i in ex], row_labels,
                            str(out / f"level_sets_{eb}.png"), args.bands)
    P.plot_pets_metrics(stats, gap, level, level_u, blocks, str(out / "metrics.png"))
    print(f"pronto: {out}/")


if __name__ == "__main__":
    main()

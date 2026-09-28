"""Por que esta imagem teve uma classificacao diferente daquela? (Oxford-IIIT Pet)

A Gram cruzada (`src/cross_pair.py`) poe as posicoes das duas imagens no mesmo espaco: para
cada posicao de A, a distancia a posicao mais proxima de B. Alta = regiao de A sem
equivalente em B, isto e, o que diferencia as duas para o modelo. A distancia e angular (a
variante validada para "mesmo conceito", ver `src/level_sets.py`).

Dois tipos de par, escolhidos sem usar o metodo:

  racas confundiveis           i e j de duas racas que o modelo mais confunde (confusao suave
                               no teste), cada uma classificada corretamente;
  mesma raca, outra predicao   i e j da mesma raca, i classificada corretamente e j nao: o
                               caso de analise de erro ("por que esta errou e aquela nao?").

Teste: apagar em A as posicoes sem correspondente deve derrubar a margem de decisao
logit[pred A] - logit[pred B] em A mais rapido do que apagar as posicoes com correspondente
(gap). A distancia entre caracteristicas nao serve: apagar qualquer parte do animal afasta A
de B, e um mapa que so marca o centro ganha tanto quanto o metodo. Comparacoes: o mesmo mapa na
rede nao treinada, a ordem aleatoria, o centro, o Grad-CAM contrastivo (gradiente da mesma
margem, a referencia usual para "por que c_a e nao c_b") e, para a especificidade, o mapa de A
contra uma terceira imagem C de outra raca, avaliado na margem de A contra B: se o mapa diz o
que difere de B, e nao so o que importa em A, ele tem de ganhar deste. A mascara diz se as
regioes apontadas estao no animal ou no fundo.

Consulta de regiao (`--query`): fixada uma posicao de uma imagem (por exemplo o olho), onde ela
casa em outras imagens, com z (destaque do melhor correspondente) e reciprocidade. `--grid`
desenha a grade numerada para escolher a posicao.

Uso:
    python pets_differences.py --pairs 20
    python pets_differences.py --grid
    python pets_differences.py --query 12:9,14 --against 3 --same-class
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src import plots_cross_pair as PC
from src import plots_region as PR
from src.cross_pair import correspondence, cross_gram, distance_matrix, level_from, reference_position, separation_curves
from src.data_pets import PetsDataset, load_pets
from src.level_sets import center_prior, mask_to_grid
from src.pixel_energy import extract_pixel_sources, gradcam
from src.report import clean, fmt, table
from src.train_pets import load_or_train, models_from

ROLES = ("raças confundíveis", "mesma raça, outra predição")


@torch.no_grad()
def predict(model, x: torch.Tensor, device, batch: int = 128) -> np.ndarray:
    return np.concatenate([torch.softmax(model(x[s : s + batch].to(device)), 1).cpu().numpy()
                           for s in range(0, len(x), batch)])


def pick_pairs(labels, preds, probs, count: int, seed: int) -> list[dict]:
    """Metade dos pares entre as racas mais confundidas, metade dentro da mesma raca."""
    rng = np.random.default_rng(seed)
    n_classes = probs.shape[1]
    soft = np.zeros((n_classes, n_classes))
    for a in range(n_classes):
        if (labels == a).any():
            soft[a] = probs[labels == a].mean(0)
    soft = soft + soft.T
    np.fill_diagonal(soft, 0)
    iu = np.triu_indices(n_classes, 1)
    top = np.argsort(-soft[iu])
    correct = preds == labels
    pairs = []
    for t in top:
        if len([p for p in pairs if p["role"] == ROLES[0]]) >= count // 2:
            break
        a, b = iu[0][t], iu[1][t]
        ia, ib = np.flatnonzero(correct & (labels == a)), np.flatnonzero(correct & (labels == b))
        if len(ia) and len(ib):
            pairs.append({"role": ROLES[0], "i": int(rng.choice(ia)), "j": int(rng.choice(ib))})
    for c in rng.permutation(n_classes):
        if len([p for p in pairs if p["role"] == ROLES[1]]) >= count // 2:
            break
        ok, wrong = np.flatnonzero(correct & (labels == c)), np.flatnonzero(~correct & (labels == c))
        if len(ok) and len(wrong):
            pairs.append({"role": ROLES[1], "i": int(rng.choice(ok)), "j": int(rng.choice(wrong))})
    return pairs


def pick_third(labels, preds, pairs, seed: int) -> None:
    """Para cada par, uma imagem C correta de uma raca fora de {rotulos e predicoes de A e B}."""
    rng = np.random.default_rng(seed + 1)
    correct = np.flatnonzero(preds == labels)
    for pair in pairs:
        used = {labels[pair["i"]], labels[pair["j"]], preds[pair["i"]], preds[pair["j"]]}
        pair["k"] = int(rng.choice([c for c in correct if labels[c] not in used]))


def contrastive_gradcam(model, x: torch.Tensor, block: str, ca: int, cb: int, device) -> np.ndarray:
    """Grad-CAM da margem logit[c_a] - logit[c_b] no estado de `block`."""
    store = {}

    def fn(_mod, _inputs, output):
        output.retain_grad()
        store["h"] = output

    handle = model.get_block(block).register_forward_hook(fn)
    try:
        with torch.enable_grad():
            logits = model(x[None].to(device))
            model.zero_grad(set_to_none=True)
            (logits[0, ca] - logits[0, cb]).backward()
        h = store["h"]
        cam = gradcam(h.detach().float().cpu(), h.grad.detach().float().cpu())[0].numpy()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
    return cam


def field(src, i: int, j: int, block: str, device, args):
    g, p, h, w = cross_gram(src.acts[block][i : i + 1], src.grads[block][i : i + 1],
                            src.acts[block][j : j + 1], src.grads[block][j : j + 1],
                            src.metric[block], device, args.max_side, "trace_center", 1.0, args.floor)
    return correspondence(g, p, h, w), p, h, w


def mask_inside(f: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    m = mask_to_grid(mask, *f.shape)
    f = np.maximum(f - f.min(), 0.0)  # a distancia minima vira "excesso" sobre o mais proximo
    return float((f * m).sum() / max(f.sum(), 1e-12)), float(m.mean())


COMPARE = {
    "gap": "sem correspondente em B (método)",
    "gap_gradcam": "Grad-CAM contrastivo",
    "gap_other": "sem correspondente em C (especificidade)",
    "gap_cross": "piso (rede não treinada)",
    "gap_center": "só o centro",
    "gap_random": "ordem aleatória",
}


def run_pairs(args, device, blob, data, model, untrained, test, x_all, out: Path) -> None:
    probs = predict(model, x_all, device)
    labels = data["labels"][test]
    preds = probs.argmax(1)
    pairs = pick_pairs(labels, preds, probs, args.pairs, args.seed)
    pick_third(labels, preds, pairs, args.seed)
    involved = sorted({p[key] for p in pairs for key in ("i", "j", "k")})
    local = {g: k for k, g in enumerate(involved)}
    x = x_all[involved]
    blocks = model.block_names
    src = extract_pixel_sources(model, x, labels[involved], blocks, device, 16, "pred")
    src_u = extract_pixel_sources(untrained, x, labels[involved], blocks, device, 16, "pred")
    masks = data["masks"][test]
    rng = np.random.default_rng(args.seed)
    results = {b: [] for b in blocks}
    examples = []
    for b in blocks:
        for pair in pairs:
            i, j, k = local[pair["i"]], local[pair["j"]], local[pair["k"]]
            ca, cb = int(preds[pair["i"]]), int(preds[pair["j"]])
            corr, p, h, w = field(src, i, j, b, device, args)
            corr_u, *_ = field(src_u, i, j, b, device, args)
            corr_c, *_ = field(src, i, k, b, device, args)
            fa = corr["no_counterpart_a"]

            def gap(f, key="gap_margin"):
                return separation_curves(model, x[i], x[j], f, device, args.steps, args.seed, (ca, cb))[key]

            curve = separation_curves(model, x[i], x[j], fa, device, args.steps, args.seed, (ca, cb))
            cam = contrastive_gradcam(model, x[i], b, ca, cb, device)
            row = {
                "role": pair["role"], "clean_margin": curve["clean_margin"],
                "gap": curve["gap_margin"], "gap_features": curve["gap"],
                "gap_gradcam": gap(cam), "gap_other": gap(corr_c["no_counterpart_a"]),
                "gap_cross": gap(corr_u["no_counterpart_a"]), "gap_center": gap(center_prior(h, w)),
                "gap_random": gap(rng.random((h, w))),
            }
            row["inside"], row["chance"] = mask_inside(fa, masks[pair["i"]])
            row["inside_center"], _ = mask_inside(center_prior(h, w), masks[pair["i"]])
            row["inside_gradcam"], _ = mask_inside(cam, masks[pair["i"]])
            row["beats_other"] = float(row["gap"] > row["gap_other"])
            row["beats_center"] = float(row["gap"] > row["gap_center"])
            row["beats_gradcam"] = float(row["gap"] > row["gap_gradcam"])
            results[b].append(row)
            if b == args.example_block and len(examples) < args.examples:
                s0 = reference_position(fa)
                pa, pb = level_from(corr["distance"], s0, p, h, w)
                examples.append({**pair, "no_counterpart_a": fa, "no_counterpart_b": corr["no_counterpart_b"],
                                 "p_a": pa, "p_b": pb, "s0": divmod(s0, w), "gap": row["gap"], "curves": curve})
        med = lambda key: float(np.median([r[key] for r in results[b]]))
        print(f"  {b}: gap da margem {med('gap'):.3f} (Grad-CAM {med('gap_gradcam'):.3f}, contra C "
              f"{med('gap_other'):.3f}, piso {med('gap_cross'):.3f}, centro {med('gap_center'):.3f}, "
              f"aleatório {med('gap_random'):.3f}); no animal {med('inside'):.3f} (acaso {med('chance'):.3f})",
              flush=True)

    def stat(b, key, role=None, agg=np.median):
        return float(agg([r[key] for r in results[b] if role in (None, r["role"])]))

    keys = [k for k in results[blocks[0]][0] if k != "role"]
    summary = {b: {k: stat(b, k, agg=np.mean if k.startswith("beats") else np.median) for k in keys}
               | {f"gap_{role}": stat(b, "gap", role) for role in ROLES}
               for b in blocks}
    (out / "report.json").write_text(json.dumps(clean({"config": vars(args), "pairs": pairs, "summary": summary,
                                                       "rows": results}), indent=2, ensure_ascii=False))
    classes = [str(c).replace("_", " ") for c in data["classes"]]
    md = [f"# Por que esta imagem teve outra classificação? — Oxford-IIIT Pet ({len(pairs)} pares)\n",
          "Medianas sobre os pares. Gap: área da margem logit[pred A] − logit[pred B] em A apagando primeiro o que "
          "**tem** correspondente menos a área apagando primeiro o que **não tem** (maior = o mapa aponta o que "
          "sustenta a decisão de A contra a de B). A especificidade usa o mapa de A contra uma terceira imagem C, "
          "de outra raça, avaliado na mesma margem.\n",
          table("Gap da margem de decisão", {
              **{COMPARE["gap"]: {b: summary[b]["gap"] for b in blocks}},
              **{f"  {role}": {b: summary[b][f"gap_{role}"] for b in blocks} for role in ROLES},
              **{label: {b: summary[b][key] for b in blocks} for key, label in COMPARE.items() if key != "gap"}},
              blocks),
          table("Fração dos pares em que o método vence (estrito)", {
              "vs. mapa contra C (especificidade)": {b: summary[b]["beats_other"] for b in blocks},
              "vs. só o centro": {b: summary[b]["beats_center"] for b in blocks},
              "vs. Grad-CAM contrastivo": {b: summary[b]["beats_gradcam"] for b in blocks}}, blocks),
          table("Onde está a diferença: fração do mapa sobre o animal", {
              "regiões sem correspondente": {b: summary[b]["inside"] for b in blocks},
              "Grad-CAM contrastivo": {b: summary[b]["inside_gradcam"] for b in blocks},
              "só o centro": {b: summary[b]["inside_center"] for b in blocks},
              "acaso (fração da imagem ocupada pelo animal)": {b: summary[b]["chance"] for b in blocks}}, blocks),
          table("Leitura antiga (distância de cosseno entre características; não responde à pergunta)", {
              "sem correspondente em B": {b: summary[b]["gap_features"] for b in blocks}}, blocks),
          "\n## Pares ilustrados\n"]
    for e in examples:
        md.append(f"- {e['role']}: A = {classes[labels[e['i']]]} (predita {classes[preds[e['i']]]}), "
                  f"B = {classes[labels[e['j']]]} (predita {classes[preds[e['j']]]}), gap {fmt(e['gap'])}")
    (out / "tables.md").write_text("\n".join(md) + "\n")

    images = data["images"][test]
    PC.plot_cross_pairs(images, labels, preds, classes, examples, args.example_block, args.bands,
                        str(out / f"pairs_{args.example_block}.png"))
    PC.plot_cross_tests(summary, blocks, ROLES, str(out / "gap_by_block.png"))


def run_query(args, device, blob, data, model, test, x_all, out: Path) -> None:
    labels = data["labels"][test]
    probs = predict(model, x_all, device)
    preds = probs.argmax(1)
    correct = preds == labels
    block = args.example_block
    if args.grid:
        chosen = list(np.flatnonzero(correct)[: args.grid_images])
        src = extract_pixel_sources(model, x_all[chosen[:1]], labels[chosen[:1]], [block], device, 1, "pred")
        grid = min(src.acts[block].shape[-1], args.max_side)
        PR.plot_grid(data["images"][test][chosen], [f"{data['classes'][labels[i]]} · {i}" for i in chosen],
                     grid, str(out / "grid.png"))
        print(f"grade {grid}x{grid} em {out}/grid.png; imagens {chosen}")
        return
    rng = np.random.default_rng(args.seed)
    results = []
    for text in args.query:
        qi, cell = text.split(":")
        qi, (qr, qc) = int(qi), map(int, cell.split(","))
        pool = np.flatnonzero(correct & ((labels == labels[qi]) if args.same_class else (labels != labels[qi])))
        pool = pool[pool != qi]
        targets = [int(t) for t in rng.permutation(pool)[: args.against]]
        order = [qi, *targets]
        src = extract_pixel_sources(model, x_all[order], labels[order], [block], device, 8, "pred")
        matches = []
        for k, t in enumerate(targets, start=1):
            g, p, h, w = cross_gram(src.acts[block][0:1], src.grads[block][0:1], src.acts[block][k : k + 1],
                                    src.grads[block][k : k + 1], src.metric[block], device, args.max_side,
                                    "trace_center", 1.0, args.floor)
            d = distance_matrix(g)
            s0 = qr * w + qc
            to_t = d[s0, p:]
            best = int(to_t.argmin())
            matches.append({"target": t, "field": to_t.reshape(h, w), "best": divmod(best, w),
                            "z": float((to_t[best] - np.median(to_t)) / max(to_t.std(), 1e-12)),
                            "reciprocal": bool(int(d[p + best, :p].argmin()) == s0)})
        results.append({"query": (qi, qr, qc), "grid": (h, w), "matches": matches})
        print(f"consulta {text}: z médio {np.mean([m['z'] for m in matches]):+.2f}, "
              f"recíproco em {np.mean([m['reciprocal'] for m in matches]):.0%}")
    tag = "same" if args.same_class else "other"
    PR.plot_matches(data["images"][test], labels, [str(c).replace("_", " ") for c in data["classes"]],
                    results, args.window, str(out / f"query_{tag}.png"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--max-side", type=int, default=32)
    ap.add_argument("--floor", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--example-block", default="b4")
    ap.add_argument("--bands", type=int, default=6)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--grid-images", type=int, default=6)
    ap.add_argument("--query", action="append", default=[], help="imagem:linha,coluna (índices do teste)")
    ap.add_argument("--against", type=int, default=6)
    ap.add_argument("--same-class", action="store_true")
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--display-gamma", type=float, default=0.5)
    ap.add_argument("--out", default="outputs/pets_differences")
    args = ap.parse_args()
    PC.DISPLAY_GAMMA = PR.DISPLAY_GAMMA = args.display_gamma

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = load_or_train(args.seed)
    data = load_pets(blob["size"])
    model, untrained = models_from(blob, device)
    test = np.flatnonzero(data["split"] == 1)
    ds = PetsDataset(data["images"][test], data["labels"][test], blob["mean"], blob["std"])
    x_all = torch.stack([ds[i][0] for i in range(len(ds))])
    if args.grid or args.query:
        run_query(args, device, blob, data, model, test, x_all, out)
    else:
        run_pairs(args, device, blob, data, model, untrained, test, x_all, out)
    print(f"pronto: {out}/")


if __name__ == "__main__":
    main()

"""Oxford-IIIT Pet: 37 racas de gatos e cachorros, com mascara de segmentacao por imagem.

A base fica fora do repositorio, em `~/Desktop/torch/data/oxford-iiit-pet` (`images/`,
`annotations/`), como os dados brutos. Cada imagem e reduzida pelo lado menor e recortada no
centro em `size` x `size`; a mascara (trimap: 1 = animal, 2 = fundo, 3 = borda) recebe o mesmo
recorte, com interpolacao do vizinho mais proximo, e a borda conta como animal. O resultado
vai para um `.npz` em `outputs/` e as execucoes seguintes nao tocam mais nos JPGs.

A tarefa sao as **37 racas**, e nao gato × cachorro: com duas classes a margem contrastiva se
reduz a diferenca entre dois logits e varias leituras viram tautologia. A particao e a
oficial da base (`trainval.txt`, `test.txt`), de modo que o teste nunca foi visto no treino.
A mascara nunca entra no treino: e so a verdade externa para conferir o que o metodo aponta.
"""

from __future__ import annotations

import math
import random
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path.home() / "Desktop" / "torch" / "data" / "oxford-iiit-pet"
SIZE = 128


def _read_split(root: Path, name: str) -> list[tuple[str, int, int]]:
    """(nome, classe 0..36, especie 0 = gato / 1 = cachorro) de uma lista oficial."""
    out = []
    for line in (root / "annotations" / f"{name}.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        stem, cls, species, _ = line.split()
        out.append((stem, int(cls) - 1, int(species) - 1))
    return out


def _center_square(im: Image.Image, size: int, resample) -> Image.Image:
    w, h = im.size
    scale = size / min(w, h)
    im = im.resize((max(size, round(w * scale)), max(size, round(h * scale))), resample)
    w, h = im.size
    left, top = (w - size) // 2, (h - size) // 2
    return im.crop((left, top, left + size, top + size))


def _load_one(args: tuple[str, str, int]):
    image_path, mask_path, size = args
    warnings.filterwarnings("ignore")
    try:
        with Image.open(image_path) as im:
            img = np.asarray(_center_square(im.convert("RGB"), size, Image.BICUBIC), dtype=np.uint8)
        with Image.open(mask_path) as m:
            tri = np.asarray(_center_square(m, size, Image.NEAREST), dtype=np.uint8)
        return img, (tri != 2).astype(np.uint8)  # animal e borda = 1, fundo = 0
    except Exception:
        return None


def build_cache(root: Path = ROOT, size: int = SIZE, cache: Path | None = None) -> Path:
    cache = cache or Path("outputs") / f"pets_{size}.npz"
    rows = [(*r, 0) for r in _read_split(root, "trainval")] + [(*r, 1) for r in _read_split(root, "test")]
    jobs = [(str(root / "images" / f"{s}.jpg"), str(root / "annotations" / "trimaps" / f"{s}.png"), size)
            for s, *_ in rows]
    with ProcessPoolExecutor() as pool:
        loaded = list(pool.map(_load_one, jobs, chunksize=64))
    keep = [i for i, x in enumerate(loaded) if x is not None]
    names = {}
    for s, cls, *_ in rows:
        names.setdefault(cls, "_".join(s.split("_")[:-1]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        images=np.stack([loaded[i][0] for i in keep]),
        masks=np.stack([loaded[i][1] for i in keep]),
        labels=np.asarray([rows[i][1] for i in keep], dtype=np.int64),
        species=np.asarray([rows[i][2] for i in keep], dtype=np.int64),
        split=np.asarray([rows[i][3] for i in keep], dtype=np.int64),
        names=np.asarray([rows[i][0] for i in keep]),
        classes=np.asarray([names[c] for c in sorted(names)]),
        skipped=np.asarray([rows[i][0] for i in range(len(rows)) if loaded[i] is None]),
    )
    print(f"{len(keep)} imagens em {cache}; ignoradas: {len(rows) - len(keep)}")
    return cache


def load_pets(size: int = SIZE, cache: Path | None = None) -> dict:
    """Imagens, mascaras, rotulos, especie, particao (0 treino, 1 teste) e nomes das classes."""
    cache = cache or Path("outputs") / f"pets_{size}.npz"
    if not cache.exists():
        build_cache(size=size, cache=cache)
    blob = np.load(cache)
    return {k: blob[k] for k in blob.files}


def channel_stats(images: np.ndarray, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    total, sq, count = np.zeros(3), np.zeros(3), 0
    for s in range(0, len(images), chunk):
        x = images[s : s + chunk].astype(np.float64) / 255.0
        total += x.sum((0, 1, 2))
        sq += (x**2).sum((0, 1, 2))
        count += x.shape[0] * x.shape[1] * x.shape[2]
    mean = total / count
    return mean.astype(np.float32), np.sqrt(sq / count - mean**2).astype(np.float32)


def balanced_subset(labels: np.ndarray, per_class: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    picked = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        picked.append(np.sort(idx[:per_class]))
    return np.concatenate(picked)


def _random_resized_crop(img: np.ndarray, scale=(0.55, 1.0), ratio=(0.75, 1.333)) -> np.ndarray:
    """Recorte de area e proporcao aleatorias, reamostrado de volta ao tamanho original."""
    s = img.shape[0]
    area = s * s
    for _ in range(10):
        target = area * random.uniform(*scale)
        aspect = math.exp(random.uniform(math.log(ratio[0]), math.log(ratio[1])))
        w, h = int(round(math.sqrt(target * aspect))), int(round(math.sqrt(target / aspect)))
        if w <= s and h <= s:
            r, c = random.randint(0, s - h), random.randint(0, s - w)
            crop = img[r : r + h, c : c + w]
            t = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1)))[None]
            return F.interpolate(t, size=(s, s), mode="bilinear", align_corners=False)[0].numpy().transpose(1, 2, 0)
    return img


def _jitter(img: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """Brilho, contraste e saturacao, com a mesma amplitude nos tres."""
    img = img * random.uniform(1 - strength, 1 + strength)                      # brilho
    m = img.mean()
    img = (img - m) * random.uniform(1 - strength, 1 + strength) + m            # contraste
    grey = img.mean(axis=2, keepdims=True)
    img = grey + (img - grey) * random.uniform(1 - strength, 1 + strength)      # saturacao
    return np.clip(img, 0.0, 1.0)


class PetsDataset(Dataset):
    """Em memoria. `train=True` liga o aumento de dados.

    Com 3 312 imagens de treino para 37 racas (90 por raca), o aumento e o que separa um
    treino do zero utilizavel de um que decora: espelhamento, recorte de escala e proporcao
    aleatorias, jitter de cor e apagamento de um retangulo (Zhong et al., 2020). A avaliacao
    (`train=False`) nao tem nada disso -- e o que a analise usa.
    """

    def __init__(self, images, labels, mean, std, train: bool = False, pad: int = 12,
                 erase: float = 0.25) -> None:
        self.images, self.labels = images, labels
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
        self.train, self.pad, self.erase = train, pad, erase

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        img = self.images[i].astype(np.float32) / 255.0
        if self.train:
            if random.random() < 0.5:
                img = img[:, ::-1]
            img = _random_resized_crop(np.ascontiguousarray(img))
            if random.random() < 0.8:
                img = _jitter(img)
        img = (img - self.mean) / self.std
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        if self.train and random.random() < self.erase:
            s = x.shape[-1]
            h, w = random.randint(s // 8, s // 3), random.randint(s // 8, s // 3)
            r, c = random.randint(0, s - h), random.randint(0, s - w)
            x[:, r : r + h, c : c + w] = torch.randn(3, h, w) * 0.5
        return x, int(self.labels[i])


def make_loader(ds: Dataset, batch_size: int, shuffle: bool = False, drop_last: bool = False,
                workers: int = 8) -> DataLoader:
    """DataLoader com workers persistentes, herdando a memoria do processo pai.

    No Python 3.14 o metodo padrao de criar processos no Linux passou a ser `forkserver`, que
    serializa o dataset inteiro (o vetor de imagens em memoria) para cada worker. Com 8 workers
    isso truncou o pickle e derrubou o treino da `vit_seed2`. Com `fork` os workers herdam a
    memoria por copia sob escrita e nada e serializado; eles so usam numpy, nunca a GPU.
    """
    kw = {"num_workers": workers, "persistent_workers": workers > 0, "pin_memory": torch.cuda.is_available()}
    if workers > 0:
        kw["multiprocessing_context"] = "fork"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kw)

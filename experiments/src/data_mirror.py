"""As bases tabulares lidas das fontes originais, para quando o OpenML nao responde.

Cada base do OpenML usada no artigo e a copia de um arquivo publicado na UCI (ou, no caso da
`dna`, no StatLog via LIBSVM). Este modulo le esses originais, guardados em
`~/Desktop/torch/data/tabular/raw/`, e devolve as mesmas matrizes que `fetch_openml`:

  - os rotulos passam a texto e sao codificados pelo `LabelEncoder`, como no OpenML. Isso
    importa: a ordem lexicografica decide qual e a classe 0, que e o alvo fixo do gradiente
    (na `mfeat-factors`, por exemplo, os rotulos '1'..'10' ordenam '1', '10', '2', ...);
  - as linhas seguem a ordem dos arquivos de treino e depois de teste, e a particao do
    experimento e refeita por `data_tabular.split` de qualquer forma;
  - a forma de cada base e conferida contra a do OpenML em `EXPECTED`. A unica diferenca
    conhecida e a `satimage`: o OpenML 182 tem 6 430 linhas e os arquivos da UCI, 6 435.

Os arquivos sao baixados uma vez por `download()` (ou a mao, com os nomes de `SOURCES`) e cada
base vira um `.npz` em `~/Desktop/torch/data/tabular/`.
"""

from __future__ import annotations

import gzip
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT = Path.home() / "Desktop" / "torch" / "data" / "tabular"
RAW = ROOT / "raw"
UCI = "https://archive.ics.uci.edu/static/public"
LIBSVM = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass"
SOURCES = {
    "letter.zip": f"{UCI}/59/letter+recognition.zip",
    "pendigits.zip": f"{UCI}/81/pen+based+recognition+of+handwritten+digits.zip",
    "optdigits.zip": f"{UCI}/80/optical+recognition+of+handwritten+digits.zip",
    "satimage.zip": f"{UCI}/146/statlog+landsat+satellite.zip",
    "segment.zip": f"{UCI}/50/image+segmentation.zip",
    "har.zip": f"{UCI}/240/human+activity+recognition+using+smartphones.zip",
    "mfeat.zip": f"{UCI}/72/multiple+features.zip",
    "gas.zip": f"{UCI}/224/gas+sensor+array+drift+dataset.zip",
    "covertype.zip": f"{UCI}/31/covertype.zip",
    "dna.scale": f"{LIBSVM}/dna.scale",
    "dna.scale.t": f"{LIBSVM}/dna.scale.t",
    "titanic3.csv": "https://hbiostat.org/data/repo/titanic3.csv",
}
# (amostras, atributos, classes) no OpenML, antes da subamostragem da covertype
EXPECTED = {
    "letter": (20000, 16, 26), "pendigits": (10992, 16, 10), "optdigits": (5620, 64, 10),
    "satimage": (6430, 36, 6), "segment": (2310, 19, 7), "har": (10299, 561, 6),
    "mfeat-factors": (2000, 216, 10), "gas-drift": (13910, 128, 6), "dna": (3186, 180, 3),
    "covertype": (581012, 54, 7),
}
TOLERANCE = {"satimage": 5}
# nomes dos atributos das duas bases dos estudos de caso: os da `letter` sao os do OpenML; os da
# `pendigits` sao as coordenadas dos 8 pontos reamostrados do traco (Alpaydin & Alimoglu), na ordem
# x1, y1, ..., x8, y8 dos arquivos originais
LETTER_FEATURES = ["x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar", "y2bar",
                   "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]
PENDIGITS_FEATURES = [f"{axis}{k}" for k in range(1, 9) for axis in ("x", "y")]
NAMED = {"letter": LETTER_FEATURES, "pendigits": PENDIGITS_FEATURES}


def download(force: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        path = RAW / name
        if path.exists() and not force:
            continue
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())


def _zip_text(zname: str, member: str) -> str:
    with zipfile.ZipFile(RAW / zname) as z:
        return z.read(member).decode("latin-1")


def _csv(text: str, sep: str | None = ",") -> np.ndarray:
    return pd.read_csv(io.StringIO(text), header=None, sep=sep, engine="python").to_numpy()


def _libsvm(text: str, n_features: int, sep_label: str | None = None) -> tuple[np.ndarray, list[str]]:
    rows, labels = [], []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        head = parts[0].split(sep_label)[0] if sep_label else parts[0]
        x = np.zeros(n_features)
        for item in parts[1:]:
            k, v = item.split(":")
            x[int(k) - 1] = float(v)
        rows.append(x)
        labels.append(str(int(float(head))))
    return np.asarray(rows), labels


def _read(name: str) -> tuple[np.ndarray, list[str]]:
    if name == "letter":
        a = _csv(_zip_text("letter.zip", "letter-recognition.data"))
        return a[:, 1:].astype(float), [str(v) for v in a[:, 0]]
    if name in ("pendigits", "optdigits"):
        a = np.vstack([_csv(_zip_text(f"{name}.zip", f"{name}.{ext}")) for ext in ("tra", "tes")])
        return a[:, :-1].astype(float), [str(int(v)) for v in a[:, -1]]
    if name == "satimage":
        a = np.vstack([_csv(_zip_text("satimage.zip", f"sat.{ext}"), sep=r"\s+") for ext in ("trn", "tst")])
        return a[:, :-1].astype(float), [str(int(v)) for v in a[:, -1]]
    if name == "segment":
        rows = []
        for member in ("segmentation.data", "segmentation.test"):
            for line in _zip_text("segment.zip", member).splitlines():
                parts = line.strip().split(",")
                if len(parts) == 20 and parts[0].isalpha() and parts[0].isupper() and "-" not in parts[0]:
                    rows.append(parts)
        a = np.asarray(rows)
        return a[:, 1:].astype(float), [v.lower() for v in a[:, 0]]
    if name == "har":
        with zipfile.ZipFile(RAW / "har.zip") as outer:
            inner_name = next(n for n in outer.namelist() if n.endswith(".zip"))
            with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as z:
                def part(split, kind):
                    member = next(n for n in z.namelist() if n.endswith(f"{split}/{kind}_{split}.txt"))
                    return _csv(z.read(member).decode(), sep=r"\s+")
                X = np.vstack([part("train", "X"), part("test", "X")])
                y = np.concatenate([part("train", "y")[:, 0], part("test", "y")[:, 0]])
        return X.astype(float), [str(int(v)) for v in y]
    if name == "mfeat-factors":
        X = _csv(_zip_text("mfeat.zip", "mfeat-fac"), sep=r"\s+").astype(float)
        return X, [str(1 + i // 200) for i in range(len(X))]  # 200 por digito, na ordem, como no OpenML
    if name == "gas-drift":
        X, y = [], []
        with zipfile.ZipFile(RAW / "gas.zip") as z:
            for b in range(1, 11):
                xb, yb = _libsvm(z.read(f"Dataset/batch{b}.dat").decode(), 128, sep_label=";")
                X.append(xb)
                y += yb
        return np.vstack(X), y
    if name == "dna":
        parts = [_libsvm((RAW / f).read_text(), 180) for f in ("dna.scale", "dna.scale.t")]
        return np.vstack([p[0] for p in parts]), parts[0][1] + parts[1][1]
    if name == "covertype":
        with zipfile.ZipFile(RAW / "covertype.zip") as z:
            a = _csv(gzip.decompress(z.read("covtype.data.gz")).decode())
        return a[:, :-1].astype(float), [str(int(v)) for v in a[:, -1]]
    raise KeyError(name)


def load(name: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """(X float64, y int64, nomes dos atributos, nomes das classes), com cache em `.npz`."""
    cache = ROOT / f"{name}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        features = NAMED.get(name) or [str(f) for f in z["features"]]
        return z["X"], z["y"], list(features), [str(c) for c in z["classes"]]
    if not RAW.exists() or not any(RAW.iterdir()):
        download()
    X, labels = _read(name)
    enc = LabelEncoder().fit(np.asarray(labels))
    y = enc.transform(np.asarray(labels)).astype(np.int64)
    n, f, k = EXPECTED[name]
    if X.shape[1] != f or len(enc.classes_) != k or abs(len(X) - n) > TOLERANCE.get(name, 0):
        raise ValueError(f"{name}: forma {X.shape}, {len(enc.classes_)} classes; esperado {EXPECTED[name]}")
    features = NAMED.get(name) or [f"V{i + 1}" for i in range(f)]
    classes = [str(c) for c in enc.classes_]
    ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y, features=np.asarray(features), classes=np.asarray(classes))
    return X, y, features, classes


def titanic_frame() -> pd.DataFrame:
    """O `titanic3` de Vanderbilt, o mesmo arquivo que o OpenML 40945 publica."""
    path = RAW / "titanic3.csv"
    if not path.exists():
        download()
    return pd.read_csv(path)

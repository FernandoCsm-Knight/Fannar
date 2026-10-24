"""Bases tabulares do OpenML: as vinte da suite de estabilidade e as dos estudos de caso.

Regra de selecao: todas as bases multiclasse do OpenML-CC18 (estudo 99) com pelo menos 2 000
amostras, menos as duplicatas (as outras cinco `mfeat-*` descrevem os mesmos 2 000 digitos da
`mfeat-factors`; `splice` sao os mesmos dados da `dna`) e a `CIFAR_10`, base de imagens ja coberta
pelas suites de imagem. Sao 18 do CC18, mais `gas-drift` e `covertype`, que vinham da primeira lista.

Criterios da primeira lista, medidos antes de fixa-la:

  multiclasse    com duas classes a margem contrastiva se reduz a diferenca entre dois
                 logits e troca de sinal com a predicao em toda a rede, e varias leituras do
                 artigo viram tautologia (Subsecao `subsec:dados`);
  tamanho        pelo menos ~2 000 amostras, para caberem treino, validacao, teste e a amostra
                 de 500 objetos da Gram vindo so do teste;
  equilibrio     nenhuma classe com poucos exemplos (por isso `wine-quality`, com uma classe de
                 5 amostras, ficou de fora; `vehicle` e `cnae-9` sao pequenas demais).

As bases com mais de 30 000 amostras sao subamostradas de forma estratificada para 30 000.
Atributos categoricos binarios viram 0/1, e os de mais niveis (`connect-4`) viram indicadores.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATASETS = {
    "letter": 6,
    "pendigits": 32,
    "optdigits": 28,
    "satimage": 182,
    "segment": 40984,
    "har": 1478,
    "mfeat-factors": 12,
    "gas-drift": 1476,
    "dna": 40670,
    "covertype": 1596,
    "isolet": 300,
    "texture": 40499,
    "first-order": 1475,
    "gesture": 4538,
    "wall-robot": 1497,
    "connect-4": 40668,
    "jungle-chess": 41027,
    "mnist": 554,
    "fashion-mnist": 40996,
    "devnagari": 40923,
}
MAX_SAMPLES = {name: 30000 for name in DATASETS}

# Base fora da suite: o HELOC (FICO, 2018), estudo de caso de P6 junto com a letter (`src/case.py`).
EXTRA = {"heloc": 46932}
ALL = DATASETS | EXTRA

# HELOC: valores especiais do dicionario da FICO. -9 = sem registro de bureau (as 588 linhas em
# que todos os atributos sao -9 saem); -8 = sem conta ou consulta valida; -7 = a condicao nunca
# ocorreu, e so aparece em atributos "meses desde" (MSinceMostRecentDelq,
# MSinceMostRecentInqexcl7days), em que "nunca" e mais antigo que qualquer ocorrencia: vira o maximo
# observado + 1. -8 e os -9 restantes viram a mediana dos valores validos. Indicadores de ausencia
# dobrariam o custo exato (2^F) a cada coluna, entao os 23 atributos ficam como estao.


@functools.lru_cache(maxsize=1)
def _heloc() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    bunch = fetch_openml(data_id=EXTRA["heloc"], as_frame=True, parser="auto")
    df = bunch.data.astype(float)
    keep = ~(df == -9).all(1).to_numpy()
    df, target = df[keep].copy(), np.asarray(bunch.target)[keep]
    for c in df.columns:
        col = df[c]
        valid = col[col >= 0]
        col = col.where(col != -7, valid.max() + 1)
        df[c] = col.where(col >= 0, valid.median())
    classes = ["Bad", "Good"]
    y = np.array([classes.index(str(t)) for t in target], dtype=np.int64)
    return df.to_numpy(), y, list(df.columns), classes


@functools.lru_cache(maxsize=2)  # as bases de imagem achatadas ocupam centenas de MB
def _openml(name: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """(X, y, nomes dos atributos, nomes das classes), com os categoricos codificados."""
    import pandas as pd
    bunch = fetch_openml(data_id=ALL[name], as_frame=True, parser="auto")
    cols, names = [], []
    for c in bunch.data.columns:
        col = bunch.data[c]
        if str(col.dtype) in ("category", "object"):
            levels = sorted(col.astype(str).unique())
            if len(levels) <= 2:
                cols.append((col.astype(str) == levels[-1]).astype(float).to_numpy())
                names.append(str(c))
            else:
                for lv in levels:
                    cols.append((col.astype(str) == lv).astype(float).to_numpy())
                    names.append(f"{c}={lv}")
        else:
            cols.append(pd.to_numeric(col, errors="coerce").to_numpy(dtype=float))
            names.append(str(c))
    enc = LabelEncoder().fit(np.asarray(bunch.target).astype(str))
    y = enc.transform(np.asarray(bunch.target).astype(str)).astype(np.int64)
    return np.column_stack(cols), y, names, [str(c) for c in enc.classes_]


def feature_names(name: str) -> list[str]:
    """Nomes dos atributos, na ordem das colunas de `load_tabular`."""
    warnings.filterwarnings("ignore")
    if name == "heloc":
        return _heloc()[2]
    return _openml(name)[2]


def class_names(name: str) -> list[str]:
    """Nomes das classes, na ordem dos codigos inteiros de `load_tabular`."""
    warnings.filterwarnings("ignore")
    if name == "heloc":
        return _heloc()[3]
    return _openml(name)[3]


def load_tabular(name: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(X float32, y int64) de uma base do OpenML, com a subamostragem fixa por semente 0."""
    warnings.filterwarnings("ignore")
    if name == "heloc":
        X, y, _, _ = _heloc()
        return X.astype(np.float32), y
    X, y, _, _ = _openml(name)
    X = np.nan_to_num(np.asarray(X, dtype=np.float64))
    if name in MAX_SAMPLES and len(y) > MAX_SAMPLES[name]:
        keep, _ = train_test_split(np.arange(len(y)), train_size=MAX_SAMPLES[name],
                                   stratify=y, random_state=0)
        X, y = X[keep], y[keep]
    return X.astype(np.float32), y


def split(y: np.ndarray, seed: int, test: float = 0.2, val: float = 0.1):
    """Indices estratificados de treino, validacao e teste."""
    idx = np.arange(len(y))
    rest, test_idx = train_test_split(idx, test_size=test, stratify=y, random_state=seed)
    train_idx, val_idx = train_test_split(rest, test_size=val / (1 - test), stratify=y[rest],
                                          random_state=seed)
    return train_idx, val_idx, test_idx


def standardize(train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """Padroniza pelas estatisticas do treino, com piso no desvio para atributos constantes."""
    mean = train.mean(0, keepdims=True)
    std = np.maximum(train.std(0, keepdims=True), 1e-6)
    return [(a - mean) / std for a in (train, *others)]


def balanced_sample(y: np.ndarray, total: int = 500, seed: int = 0) -> np.ndarray:
    """Posicoes com o mesmo numero de amostras por classe, somando ~`total`."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, total // len(classes))
    picked = []
    for c in classes:
        members = np.flatnonzero(y == c)
        rng.shuffle(members)
        picked.append(members[:per_class])
    return np.sort(np.concatenate(picked))

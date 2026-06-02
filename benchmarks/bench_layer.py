"""Benchmark mínimo: tempo de construção e análise em função de C.

Uso: PYTHONPATH=src python benchmarks/bench_layer.py
"""

from __future__ import annotations

import time

import torch

import fannar as f


def bench(C: int, d: int = 64, repeats: int = 3) -> dict:
    times: dict[str, float] = {}
    X = torch.randn(C, d, dtype=torch.float64)

    def timed(name: str, fn) -> None:
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        times[name] = best

    K1 = f.LinearKernel()(X)
    K2 = f.CosineKernel()(X)
    K3 = f.LinearKernel()(torch.randn(C, d, dtype=torch.float64))

    timed("hadamard", lambda: f.hadamard_combine([K1, K2, K3]))
    K = f.hadamard_combine([K1, K2, K3])
    timed("distance", lambda: f.distance_matrix(K))
    timed("eigendecompose", lambda: f.eigendecompose(K))
    timed("principal_subspace", lambda: f.principal_subspace(K, tau=0.9))
    D = f.distance_matrix(K)
    timed("profile", lambda: f.explanatory_profile(K, D, gradient=torch.rand(C, dtype=torch.float64)))
    return times


def main() -> None:
    for C in (64, 256, 512, 1024):
        t = bench(C)
        line = "  ".join(f"{k}={v * 1e3:.1f}ms" for k, v in t.items())
        print(f"C={C:5d} | {line}")


if __name__ == "__main__":
    main()

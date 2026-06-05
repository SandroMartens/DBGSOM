"""Benchmark: cosine BMU search variants.

Compares four implementations of argmin-cosine over an (n, d) x (m, d) search:
  1. numba  — fused prange dot+argmax, no n×m alloc
  2. numpy  — X @ W.T, BLAS DGEMM + argmax
  3. linear_kernel — sklearn wrapper around BLAS DGEMM
  4. cosine_similarity — sklearn, re-normalises (redundant on pre-normalised data)

All variants return (distances, winners) matching the BaseSom contract.
First call warms up numba JIT before timing.
"""

import time

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import normalize

from dbgsom.BaseSom import numba_find_winners_cosine

REPS = 5


def bmu_numpy(X, W):
    sim = X @ W.T
    winners = np.argmax(sim, axis=1)
    distances = 1.0 - sim[np.arange(len(X)), winners]
    return distances, winners


def bmu_linear_kernel(X, W):
    sim = linear_kernel(X, W)
    winners = np.argmax(sim, axis=1)
    distances = 1.0 - sim[np.arange(len(X)), winners]
    return distances, winners


def bmu_cosine_similarity(X, W):
    sim = cosine_similarity(X, W)
    winners = np.argmax(sim, axis=1)
    distances = 1.0 - sim[np.arange(len(X)), winners]
    return distances, winners


def bench(fn, X, W, reps=REPS):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(X, W)
        times.append(time.perf_counter() - t0)
    return min(times)  # best of reps


def run(label, n, m, d, rng):
    X = normalize(rng.standard_normal((n, d)).astype(np.float64))
    W = normalize(rng.standard_normal((m, d)).astype(np.float64))

    # JIT warmup (tiny arrays)
    Xw = normalize(rng.standard_normal((4, d)).astype(np.float64))
    Ww = normalize(rng.standard_normal((4, d)).astype(np.float64))
    numba_find_winners_cosine(Xw, Ww)

    t_numba = bench(numba_find_winners_cosine, X, W)
    t_numpy = bench(bmu_numpy, X, W)
    t_lk = bench(bmu_linear_kernel, X, W)
    t_cs = bench(bmu_cosine_similarity, X, W)

    # correctness: all winners must match numpy reference
    _, ref = bmu_numpy(X, W)
    for name, fn in [
        ("numba", numba_find_winners_cosine),
        ("linear_kernel", bmu_linear_kernel),
        ("cosine_sim", bmu_cosine_similarity),
    ]:
        _, w = fn(X, W)
        assert np.array_equal(w, ref), f"{name} mismatch"

    mem_matrix_mb = n * m * 8 / 1e6  # float64 n×m matrix

    print(
        f"{label:<28} "
        f"numba={t_numba * 1e3:6.1f}ms  "
        f"numpy={t_numpy * 1e3:6.1f}ms  "
        f"linear_kernel={t_lk * 1e3:6.1f}ms  "
        f"cosine_sim={t_cs * 1e3:6.1f}ms  "
        f"n*m matrix={mem_matrix_mb:.1f}MB"
    )


def main():
    rng = np.random.default_rng(42)

    configs = [
        # label,                n,      m,   d
        ("n=1k  m=20  d=64", 1_000, 20, 64),
        ("n=10k m=20  d=64", 10_000, 20, 64),
        ("n=10k m=100 d=64", 10_000, 100, 64),
        ("n=50k m=20  d=20", 50_000, 20, 20),
        ("n=50k m=200 d=20", 50_000, 200, 20),
        ("n=10k m=50  d=784", 10_000, 50, 784),
    ]

    print(
        f"{'config':<28} {'numba':>12} {'numpy':>12} {'linear_kernel':>19} {'cosine_sim':>16} {'peak alloc':>14}"
    )
    print("-" * 115)
    for label, n, m, d in configs:
        run(label, n, m, d, rng)


if __name__ == "__main__":
    main()

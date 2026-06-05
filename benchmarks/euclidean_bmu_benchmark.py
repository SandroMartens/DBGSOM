"""Benchmark: Euclidean BMU search — Numba vs BLAS euclidean_distances.

Reconstructs the removed numba_find_winners kernel from commit ed599b0
to verify the BLAS replacement was the right call.
"""

import time

import numba as nb
import numpy as np
import numpy.typing as npt
from sklearn.metrics.pairwise import euclidean_distances

REPS = 5


@nb.njit(cache=True, parallel=True, fastmath=True)
def numba_find_winners(
    data: npt.NDArray, weights: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """Reconstructed from commit ed599b0 (removed in that commit)."""
    n_samples = data.shape[0]
    n_features = data.shape[1]
    n_neurons = weights.shape[0]
    winners = np.empty(n_samples, dtype=np.int64)
    distances = np.empty(n_samples, dtype=data.dtype)
    for i in nb.prange(n_samples):
        best_dist_sq = np.inf
        best_j = 0
        for j in range(n_neurons):
            d_sq = 0.0
            for k in range(n_features):
                diff = data[i, k] - weights[j, k]
                d_sq += diff * diff
            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                best_j = j
        winners[i] = best_j
        distances[i] = np.sqrt(best_dist_sq)
    return distances, winners


def bmu_blas(X, W):
    D = euclidean_distances(X, W)
    winners = np.argmin(D, axis=1)
    distances = D[np.arange(len(X)), winners]
    return distances, winners


def bench(fn, X, W, reps=REPS):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(X, W)
        times.append(time.perf_counter() - t0)
    return min(times)


def run(label, n, m, d, rng):
    X = rng.standard_normal((n, d)).astype(np.float64)
    W = rng.standard_normal((m, d)).astype(np.float64)

    # JIT warmup
    Xw = rng.standard_normal((4, d)).astype(np.float64)
    Ww = rng.standard_normal((4, d)).astype(np.float64)
    numba_find_winners(Xw, Ww)

    t_numba = bench(numba_find_winners, X, W)
    t_blas = bench(bmu_blas, X, W)

    # correctness
    _, ref = bmu_blas(X, W)
    _, w = numba_find_winners(X, W)
    assert np.array_equal(w, ref), "winner mismatch"

    speedup = t_numba / t_blas
    mem_mb = n * m * 8 / 1e6

    print(
        f"{label:<28} "
        f"numba={t_numba * 1e3:7.1f}ms  "
        f"blas={t_blas * 1e3:7.1f}ms  "
        f"speedup(BLAS)={speedup:5.2f}x  "
        f"n*m={mem_mb:.1f}MB"
    )


def main():
    rng = np.random.default_rng(42)

    configs = [
        ("n=1k  m=20  d=64", 1_000, 20, 64),
        ("n=10k m=20  d=64", 10_000, 20, 64),
        ("n=10k m=100 d=64", 10_000, 100, 64),
        ("n=50k m=20  d=20", 50_000, 20, 20),
        ("n=50k m=200 d=20", 50_000, 200, 20),
        ("n=10k m=50  d=784", 10_000, 50, 784),
    ]

    print(
        f"{'config':<28} {'numba':>14} {'blas':>13} {'speedup(BLAS)':>17} {'peak alloc':>12}"
    )
    print("-" * 95)
    for label, n, m, d in configs:
        run(label, n, m, d, rng)
    print()
    print("speedup(BLAS) > 1 means BLAS is faster than Numba")


if __name__ == "__main__":
    main()

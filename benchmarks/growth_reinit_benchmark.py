"""Benchmark: full BMU search vs. pointer search after map growth.

NetworkX preserves insertion order (verified: after[:K_old] == before always).
Therefore _prev_winners indices remain valid after growth — only new neurons
are appended. Removing `_prev_winners = None` after growth and relying on the
rebuilt _neighbor_matrix would avoid the full BMU search entirely.

This benchmark measures the per-growth-event cost difference and estimates
total training overhead savings for typical growth trajectories.

Typical DBGSOM scenario:
  - Coarse phase: map grows from ~4 to K_final neurons
  - Growth events: O(K_final) total (one neuron added per event)
  - Each event: N samples need new winner initialisation
"""

import time

import numpy as np

from dbgsom._kernels import numba_find_winners_euclidean, numba_find_winners_pointer

REPS = 20
RNG = np.random.default_rng(0)

SCENARIOS = [
    ("small  (N=1k,  D=64,  K=50)", 1_000, 64, 50),
    ("medium (N=10k, D=128, K=100)", 10_000, 128, 100),
    ("large  (N=60k, D=784, K=200)", 60_000, 784, 200),
    ("large  (N=60k, D=784, K=500)", 60_000, 784, 500),
]

# growth_events ≈ K_final (one neuron added per event, starting from 4)
GROWTH_EVENTS_FACTOR = 1.0


def make_neighbor_matrix(k: int, avg_degree: int = 4) -> np.ndarray:
    mat = np.full((k, avg_degree), -1, dtype=np.int64)
    for i in range(k):
        nbrs = []
        for delta in [-1, 1, -int(k**0.5), int(k**0.5)]:
            j = i + delta
            if 0 <= j < k and j != i:
                nbrs.append(j)
            if len(nbrs) >= avg_degree:
                break
        mat[i, : len(nbrs)] = nbrs
    return mat


def bench(fn, *args):
    fn(*args)  # warmup
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn(*args)
    return (time.perf_counter() - t0) / REPS * 1000  # ms


print(
    f"{'scenario':38} | {'full (ms)':>9} | {'ptr (ms)':>8} | {'ratio':>6} | {'G events':>9} | {'saved (s)':>9}"
)
print("-" * 95)

for label, N, D, K in SCENARIOS:
    data = RNG.standard_normal((N, D)).astype(np.float64)
    weights = RNG.standard_normal((K, D)).astype(np.float64)
    prev_winners = RNG.integers(0, K, size=N).astype(np.int64)
    neighbor_matrix = make_neighbor_matrix(K)

    t_full = bench(numba_find_winners_euclidean, data, weights)
    t_ptr = bench(
        numba_find_winners_pointer, data, weights, prev_winners, neighbor_matrix
    )

    ratio = t_full / t_ptr
    growth_events = int(K * GROWTH_EVENTS_FACTOR)
    saved_s = growth_events * (t_full - t_ptr) / 1000

    print(
        f"{label:38} | {t_full:>9.1f} | {t_ptr:>8.1f} | {ratio:>6.1f}× | {growth_events:>9} | {saved_s:>9.2f}"
    )

"""Benchmark: _build_neighbor_matrix — old (Floyd-Warshall scan) vs new (adjacency dict).

Measures wall time for both implementations across several map sizes.
Also verifies that both produce identical neighbor matrices.
"""

import time

import networkx as nx
import numpy as np

from dbgsom.SomVQ import SomVQ

RNG = np.random.default_rng(0)
REPS = 20


def make_grid_som(rows: int, cols: int) -> SomVQ:
    som = SomVQ()
    som.som_ = nx.grid_2d_graph(rows, cols)
    for node in som.som_.nodes:
        som.som_.nodes[node]["weight"] = RNG.standard_normal(32)
    som.neurons_ = list(som.som_.nodes)
    som._distance_matrix = nx.floyd_warshall_numpy(som.som_)
    return som


def build_neighbor_matrix_old(som: SomVQ) -> np.ndarray:
    K = len(som.neurons_)
    rows = []
    for i in range(K):
        nbrs = np.where((som._distance_matrix[i] <= 1) & (som._distance_matrix[i] > 0))[
            0
        ].astype(np.int64)
        rows.append(nbrs)
    max_len = max(len(n) for n in rows) if rows else 1
    mat = np.full((K, max_len), -1, dtype=np.int64)
    for i, nbrs in enumerate(rows):
        mat[i, : len(nbrs)] = nbrs
    return mat


def build_neighbor_matrix_new(som: SomVQ) -> np.ndarray:
    node_to_idx = {node: i for i, node in enumerate(som.neurons_)}
    rows = [
        np.array([node_to_idx[nb] for nb in som.som_.neighbors(node)], dtype=np.int64)
        for node in som.neurons_
    ]
    max_len = max(len(r) for r in rows) if rows else 1
    mat = np.full((len(som.neurons_), max_len), -1, dtype=np.int64)
    for i, nbrs in enumerate(rows):
        mat[i, : len(nbrs)] = nbrs
    return mat


def time_fn(fn, som):
    # warm-up
    fn(som)
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn(som)
        times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times)


sizes = [
    (5, 5),
    (10, 10),
    (15, 15),
    (20, 20),
    (25, 25),
    (30, 30),
]

print(
    f"{'K':>6}  {'old min':>10}  {'old mean':>10}  {'new min':>10}  {'new mean':>10}  {'speedup':>8}  {'correct':>8}"
)
print("-" * 75)

for rows, cols in sizes:
    K = rows * cols
    som = make_grid_som(rows, cols)

    t_old_min, t_old_mean = time_fn(build_neighbor_matrix_old, som)
    t_new_min, t_new_mean = time_fn(build_neighbor_matrix_new, som)

    # correctness: both matrices must contain the same neighbor sets per row
    # (column order may differ, so compare sorted rows)
    mat_old = build_neighbor_matrix_old(som)
    mat_new = build_neighbor_matrix_new(som)
    old_sets = [set(row[row >= 0].tolist()) for row in mat_old]
    new_sets = [set(row[row >= 0].tolist()) for row in mat_new]
    correct = old_sets == new_sets

    speedup = t_old_mean / t_new_mean if t_new_mean > 0 else float("inf")
    print(
        f"{K:>6}  {t_old_min * 1e6:>9.1f}µ  {t_old_mean * 1e6:>9.1f}µ"
        f"  {t_new_min * 1e6:>9.1f}µ  {t_new_mean * 1e6:>9.1f}µ"
        f"  {speedup:>7.1f}x  {'OK' if correct else 'FAIL':>8}"
    )

"""Benchmark: dense gaussian vs. dense cutgauss vs. sparse cutgauss.

Tests the _update_weights kernel (gaussian_kernel @ voronoi_set_centers)
for a ~1000-neuron map across realistic sigma values.

Results (32×32=1024 neurons, 128 features, measured 2026-06-09):

    sigma | sparsity | dense ms | sparse ms | speedup
    ------+----------+----------+-----------+--------
        1 |   98.8%  |    5.4   |     0.8   |  6.6×
        2 |   96.4%  |    5.5   |     1.5   |  3.4×
        4 |   88.2%  |    5.5   |     5.0   |  1.1×  ← break-even
        8 |   63.9%  |    4.7   |    14.0   |  0.3×
       16 |   15.6%  |    4.7   |    33.3   |  0.1×

Conclusions:
- Dense cutgauss has no performance benefit over dense gaussian — BLAS
  processes all elements regardless of value.
- Sparse pays off only at sparsity > ~90% (sigma <= 2 on a 32×32 map).
- Break-even is around sigma=4 / sparsity=88%.
- A dynamic switch (sparse if sparsity > 0.9) would help the fine-tuning
  phase (small sigma) but is irrelevant for coarse training (large sigma).
"""

import time

import numpy as np
from scipy.sparse import csr_matrix

REPS = 50
MAP_ROWS, MAP_COLS = 32, 32
N_NEURONS = MAP_ROWS * MAP_COLS  # 1024
N_FEATURES = 128
SIGMAS = [1, 2, 4, 8, 16]
RNG = np.random.default_rng(0)


def make_distance_matrix(rows: int, cols: int) -> np.ndarray:
    """L1 graph distances on a 2D grid (approximates SOM distance_matrix)."""
    coords = np.array([[r, c] for r in range(rows) for c in range(cols)])
    diff = coords[:, None, :] - coords[None, :, :]
    return np.abs(diff).sum(axis=-1).astype(np.float64)


def gaussian_kernel(dist: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(dist**2) / (2 * sigma**2))


def cutgauss_kernel_dense(dist: np.ndarray, sigma: float) -> np.ndarray:
    h = np.exp(-(dist**2) / (2 * sigma**2))
    h *= dist <= 2 * sigma
    return h


def cutgauss_kernel_sparse(dist: np.ndarray, sigma: float) -> csr_matrix:
    h = cutgauss_kernel_dense(dist, sigma)
    return csr_matrix(h)


def run_kernel(kernel, neuron_activations: np.ndarray, voronoi_set_centers: np.ndarray):
    if hasattr(kernel, "multiply"):  # scipy sparse
        weighted = kernel.multiply(neuron_activations)
        numerator = weighted @ voronoi_set_centers
        denominator = np.asarray(weighted.sum(axis=1))
    else:
        weighted = kernel * neuron_activations
        numerator = weighted @ voronoi_set_centers
        denominator = weighted.sum(axis=1, keepdims=True)
    return numerator, denominator


def benchmark(kernel_fn, dist, sigma, neuron_activations, voronoi_set_centers):
    kernel = kernel_fn(dist, sigma)
    # warmup
    run_kernel(kernel, neuron_activations, voronoi_set_centers)

    t0 = time.perf_counter()
    for _ in range(REPS):
        run_kernel(kernel, neuron_activations, voronoi_set_centers)
    elapsed = (time.perf_counter() - t0) / REPS * 1000  # ms per call

    sparsity = 0.0
    if hasattr(kernel, "toarray"):
        nnz = kernel.nnz
        sparsity = 1.0 - nnz / (N_NEURONS * N_NEURONS)
    else:
        sparsity = np.mean(kernel == 0.0)

    return elapsed, sparsity


dist = make_distance_matrix(MAP_ROWS, MAP_COLS)
neuron_activations = RNG.integers(0, 50, size=N_NEURONS).astype(np.float64)
voronoi_set_centers = RNG.standard_normal((N_NEURONS, N_FEATURES)).astype(np.float64)

print(
    f"Map: {MAP_ROWS}×{MAP_COLS} = {N_NEURONS} neurons | features={N_FEATURES} | reps={REPS}\n"
)

hdr = f"{'sigma':>6} | {'variant':>18} | {'ms/call':>8} | {'sparsity':>9}"
print(hdr)
print("-" * len(hdr))

for sigma in SIGMAS:
    variants = [
        ("gaussian (dense)", lambda d, s: gaussian_kernel(d, s)),
        ("cutgauss (dense)", lambda d, s: cutgauss_kernel_dense(d, s)),
        ("cutgauss (sparse)", lambda d, s: cutgauss_kernel_sparse(d, s)),
    ]
    for label, fn in variants:
        ms, sparsity = benchmark(
            fn, dist, sigma, neuron_activations, voronoi_set_centers
        )
        print(f"{sigma:>6} | {label:>18} | {ms:>8.3f} | {sparsity:>8.1%}")
    print()

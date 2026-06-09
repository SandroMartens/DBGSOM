"""Benchmark: smoothing step — runtime scaling and quality impact.

Two measurements:
  1. Time per _smooth_weights() call as a function of K (neurons) and D (features).
  2. Quality (QE, TE) with smoothing_steps=0,1,2,3 on digits and synthetic data.
"""

import time
from statistics import mean, stdev

import numpy as np
from sklearn.datasets import load_digits, make_blobs

from dbgsom.SomVQ import SomVQ

RNG = np.random.default_rng(0)
REPS = 5


# ── 1. Runtime scaling ────────────────────────────────────────────────────────


def make_grid_som(rows: int, cols: int, d: int) -> SomVQ:
    """Build a SomVQ with a synthetic rows×cols grid — no training needed."""
    import networkx as nx

    som = SomVQ(smoothing_steps=0, smoothing_epsilon=0.5)
    som.som_ = nx.grid_2d_graph(rows, cols)
    for node in som.som_.nodes:
        som.som_.nodes[node]["weight"] = RNG.standard_normal(d)
    som.neurons_ = list(som.som_.nodes)
    som.weights_ = np.array([som.som_.nodes[n]["weight"] for n in som.neurons_])
    return som


def time_smooth(rows: int, cols: int, d: int, reps: int = 50) -> tuple[float, int]:
    som = make_grid_som(rows, cols, d)
    k = som.som_.number_of_nodes()
    som._smooth_weights()  # warmup
    t0 = time.perf_counter()
    for _ in range(reps):
        som._smooth_weights()
    return (time.perf_counter() - t0) / reps * 1000, k


print("=== 1. Runtime per smoothing step (synthetic grid) ===\n")
print(f"{'grid':>12} | {'K':>6} | {'D':>6} | {'ms/step':>8}")
print("-" * 40)

for (rows, cols), d in [
    ((4, 5), 64),
    ((7, 7), 64),
    ((10, 10), 128),
    ((15, 15), 128),
    ((23, 22), 128),
]:
    ms, k = time_smooth(rows, cols, d)
    print(f"{rows}×{cols:>2}{'':>6} | {k:>6} | {d:>6} | {ms:>8.3f}")


# ── 2. Quality impact ─────────────────────────────────────────────────────────


def run(X, steps, seed, **kwargs):
    som = SomVQ(
        n_iter=500,
        smoothing_steps=steps,
        smoothing_epsilon=0.5,
        random_state=seed,
        sigma_end=1,
        **kwargs,
    )
    som.fit(X)
    return som.quantization_error_, som.topographic_error_, som.som_.number_of_nodes()


def report(label, X, configs, **kwargs):
    print(f"\n=== {label} ===\n")
    print(
        f"{'smoothing':>10} | {'QE mean':>9} | {'QE std':>7} | {'TE mean':>9} | {'TE std':>7} | {'K':>5}"
    )
    print("-" * 60)
    for steps in configs:
        qes, tes, ks = [], [], []
        for seed in range(REPS):
            qe, te, k = run(X, steps, seed, **kwargs)
            qes.append(qe)
            tes.append(te)
            ks.append(k)
        print(
            f"{steps:>10} | {mean(qes):>9.4f} | {stdev(qes):>7.4f} |"
            f" {mean(tes):>9.4f} | {stdev(tes):>7.4f} | {mean(ks):>5.1f}"
        )


X_d, _ = load_digits(return_X_y=True)
X_d = X_d / 16.0  # normalise to [0,1]

X_s, _ = make_blobs(n_samples=20_000, n_features=16, centers=8, random_state=0)
X_s = X_s.astype(np.float64)

report("digits (N=1797, D=64)", X_d, [0, 1, 2, 3], max_neurons=50)
report("blobs  (N=5k,   D=16)", X_s, [0, 1, 2, 3], max_neurons=80)

"""Paper benchmark: training time vs. dataset size N.
DBGSOM vs MiniSom vs SuSi vs torchsom on synthetic Gaussian data.

Usage:
    uv run python paper_benchmarks/scaling.py

Output:
    paper_benchmarks/results/scaling.csv
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dbgsom import SomVQ

RANDOM_STATE = 42
N_SIZES = [500, 1000, 2000, 5000, 10000, 20000, 50000]
D = 20
RESULTS_DIR = Path(__file__).parent / "results"

DBGSOM_PARAMS = dict(
    n_iter=2000,
    lambda_=70,
    max_neurons=1000,
    sigma_end=1,
    pointer_search="all",
    neighborhood_function="cutgauss",
    cutgauss_phase="all",
    random_state=RANDOM_STATE,
)


@dataclass
class BenchResult:
    """Timing and quality metrics for one SOM benchmark run."""

    time: float
    n_nodes: int
    qe: float
    te: float | None = None


def make_data(n: int) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_STATE)
    return rng.standard_normal((n, D)).astype(np.float32)


def _qe_from_weights(X: np.ndarray, weights: np.ndarray) -> float:
    """Mean distance from each sample to its nearest prototype."""
    # weights: (k, dim)
    dists = np.linalg.norm(X[:, None, :] - weights[None, :, :], axis=2)
    return float(np.mean(np.min(dists, axis=1)))


def benchmark_dbgsom(X: np.ndarray) -> BenchResult:
    t0 = time.perf_counter()
    som = SomVQ(**DBGSOM_PARAMS)
    som.fit(X)
    elapsed = time.perf_counter() - t0
    return BenchResult(
        time=elapsed,
        n_nodes=len(som.neurons_),
        qe=float(som.calculate_quantization_error(X)),
        te=float(som.topographic_error_),
    )


def benchmark_minisom(X: np.ndarray, n_nodes: int) -> BenchResult | None:
    try:
        from minisom import MiniSom
    except ImportError:
        return None

    side = math.ceil(math.sqrt(n_nodes))
    t0 = time.perf_counter()
    som = MiniSom(
        side,
        side,
        X.shape[1],
        sigma=1.0,
        learning_rate=0.5,
        random_seed=RANDOM_STATE,
    )
    som.train(X, num_iteration=len(X) * 20, verbose=False)
    elapsed = time.perf_counter() - t0

    weights = som.get_weights().reshape(-1, X.shape[1])  # (side*side, dim)
    qe = _qe_from_weights(X, weights)
    return BenchResult(time=elapsed, n_nodes=side * side, qe=qe)


def benchmark_susi(X: np.ndarray, n_nodes: int) -> BenchResult | None:
    try:
        from susi import SOMClustering
    except ImportError:
        return None

    side = math.ceil(math.sqrt(n_nodes))
    t0 = time.perf_counter()
    som = SOMClustering(
        n_rows=side,
        n_columns=side,
        random_state=RANDOM_STATE,
        n_iter_unsupervised=1 * len(X),
    )
    som.fit(X)  # type: ignore[arg-type]
    elapsed = time.perf_counter() - t0

    bmus = np.array(som.get_bmus(X))
    bmu_weights = som.unsuper_som_[bmus[:, 0], bmus[:, 1]]
    qe = float(np.mean(np.linalg.norm(X - bmu_weights, axis=1)))
    return BenchResult(time=elapsed, n_nodes=side * side, qe=qe)


def benchmark_torchsom(X: np.ndarray, n_nodes: int) -> BenchResult | None:
    try:
        import torch
        from torchsom import SOM
    except ImportError:
        return None

    side = math.ceil(math.sqrt(n_nodes))
    X_t = torch.from_numpy(X.astype(np.float32))
    t0 = time.perf_counter()
    som = SOM(
        side,
        side,
        X.shape[1],
        device="cuda",
        learning_rate=1,
        batch_size=len(X),  # full-batch
        sigma=0.2 * np.sqrt(n_nodes),
    )
    som.fit(X_t)
    elapsed = time.perf_counter() - t0

    try:
        # torchsom stores weights as a Parameter; convert to numpy
        w = som.weights.detach().cpu().numpy().reshape(-1, X.shape[1])
        qe = _qe_from_weights(X, w)
    except Exception:
        qe = float("nan")
    return BenchResult(time=elapsed, n_nodes=side * side, qe=qe)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for n in N_SIZES:
        print(f"N={n:>6} ...", end=" ", flush=True)
        X = make_data(n)

        r_dbgsom = benchmark_dbgsom(X)
        r_minisom = benchmark_minisom(X, r_dbgsom.n_nodes)
        r_susi = benchmark_susi(X, r_dbgsom.n_nodes)
        r_torchsom = benchmark_torchsom(X, r_dbgsom.n_nodes)

        row: dict = {
            "N": n,
            "DBGSOM_time": round(r_dbgsom.time, 3),
            "DBGSOM_n_nodes": r_dbgsom.n_nodes,
            "DBGSOM_qe": round(r_dbgsom.qe, 4),
            "DBGSOM_te": round(r_dbgsom.te, 4) if r_dbgsom.te is not None else None,
        }
        competitors = [
            ("MiniSom", r_minisom),
            ("SuSi", r_susi),
            ("torchsom", r_torchsom),
        ]
        for label, r in competitors:
            if r is not None:
                row[f"{label}_time"] = round(r.time, 3)
                row[f"{label}_n_nodes"] = r.n_nodes
                row[f"{label}_qe"] = round(r.qe, 4)

        rows.append(row)
        parts = [f"DBGSOM={r_dbgsom.time:.2f}s n={r_dbgsom.n_nodes}"]
        competitors = [
            ("MiniSom", r_minisom),
            ("SuSi", r_susi),
            ("torchsom", r_torchsom),
        ]
        for label, r in competitors:
            if r is not None:
                parts.append(f"{label}={r.time:.2f}s")
        print(" | ".join(parts))

    df = pd.DataFrame(rows).set_index("N")
    out = RESULTS_DIR / "scaling.csv"
    df.to_csv(out)
    print(f"\nSaved → {out}")
    print(df.to_string())


if __name__ == "__main__":
    main()

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
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from dbgsom import SomVQ

RANDOM_STATE = 42
expo = np.arange(0, 7, step=0.5)
N_SIZES = (1000 * 2**expo).astype("int")

D = 30
RESULTS_DIR = Path(__file__).parent / "results"

DBGSOM_PARAMS = dict(
    n_iter=1500,
    lambda_=100,
    max_neurons=500,
    sigma_end=1,
    pointer_search="all",
    neighborhood_function="cutgauss",
    cutgauss_phase="all",
    winner_stability_threshold=0.02,
    smoothing_steps=1,
    random_state=RANDOM_STATE,
)

DBGSOM_TEXTBOOK_PARAMS = dict(
    n_iter=1500,
    lambda_=100,
    max_neurons=500,
    sigma_end=1,
    pointer_search="none",
    neighborhood_function="gaussian",
    cutgauss_phase="none",
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
    X, _ = make_blobs(n_samples=n, n_features=D, centers=10, random_state=RANDOM_STATE)
    return StandardScaler().fit_transform(X).astype(np.float32)


def _qe_from_weights(X: np.ndarray, weights: np.ndarray, chunk: int = 2000) -> float:
    """Mean distance from each sample to its nearest prototype (chunked to cap RAM)."""
    mins = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), chunk):
        sl = X[i : i + chunk]
        d = np.linalg.norm(sl[:, None, :] - weights[None, :, :], axis=2)
        mins[i : i + chunk] = d.min(axis=1)
    return float(mins.mean())


def benchmark_dbgsom(X: np.ndarray, params: dict | None = None) -> BenchResult:
    t0 = time.perf_counter()
    som = SomVQ(**(params or DBGSOM_PARAMS))  # type: ignore[arg-type]
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
        sigma=0.2 * np.sqrt(n_nodes),
        random_seed=RANDOM_STATE,
    )
    som.train(X, num_iteration=500 * n_nodes, verbose=False)
    # som.train(X, num_iteration=50, verbose=False, use_epochs=True)
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
        n_iter_unsupervised=500 * n_nodes,
        # n_iter_unsupervised=50,
    )
    som.fit(X)  # type: ignore[arg-type]
    elapsed = time.perf_counter() - t0

    bmus = np.array(som.get_bmus(X))
    bmu_weights = som.unsuper_som_[bmus[:, 0], bmus[:, 1]]
    qe = float(np.mean(np.linalg.norm(X - bmu_weights, axis=1)))
    return BenchResult(time=elapsed, n_nodes=side * side, qe=qe)


_torchsom_warmed_up = False


def _warmup_cuda() -> str:
    """Initialize CUDA once before benchmarking to exclude warmup from timing."""
    global _torchsom_warmed_up
    try:
        import torch

        if torch.cuda.is_available() and not _torchsom_warmed_up:
            torch.zeros(1).cuda()
            _torchsom_warmed_up = True
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def benchmark_torchsom(X: np.ndarray, n_nodes: int, device: str) -> BenchResult | None:
    try:
        import torch
        from torchsom import SOM
    except ImportError:
        return None

    if device == "cuda" and not torch.cuda.is_available():
        return None

    side = math.ceil(math.sqrt(n_nodes))
    X_t = torch.from_numpy(X.astype(np.float32))
    if device == "cuda":
        X_t = X_t.cuda()
    t0 = time.perf_counter()
    som = SOM(
        side,
        side,
        X.shape[1],
        device=device,
        epochs=30,
        batch_size=int(0.1 * len(X)),
        sigma=0.2 * np.sqrt(n_nodes),
    )
    som.fit(X_t)
    elapsed = time.perf_counter() - t0

    try:
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
        r_dbgsom_tb = benchmark_dbgsom(X, DBGSOM_TEXTBOOK_PARAMS)
        r_minisom = benchmark_minisom(X, r_dbgsom.n_nodes)
        r_susi = benchmark_susi(X, r_dbgsom.n_nodes)
        r_torchsom_cpu = benchmark_torchsom(X, r_dbgsom.n_nodes, "cpu")
        r_torchsom_gpu = benchmark_torchsom(X, r_dbgsom.n_nodes, "cuda")

        row: dict = {
            "N": n,
            "DBGSOM_time": round(r_dbgsom.time, 3),
            "DBGSOM_n_nodes": r_dbgsom.n_nodes,
            "DBGSOM_qe": round(r_dbgsom.qe, 4),
            "DBGSOM_te": round(r_dbgsom.te, 4) if r_dbgsom.te is not None else None,
            "DBGSOM_textbook_time": round(r_dbgsom_tb.time, 3),
            "DBGSOM_textbook_n_nodes": round(r_dbgsom_tb.n_nodes, 3),
            "DBGSOM_textbook_qe": round(r_dbgsom_tb.qe, 4),
        }
        competitors = [
            ("MiniSom", r_minisom),
            ("SuSi", r_susi),
            ("torchsom_cpu", r_torchsom_cpu),
            ("torchsom_gpu", r_torchsom_gpu),
        ]
        for label, r in competitors:
            if r is not None:
                row[f"{label}_time"] = round(r.time, 3)
                row[f"{label}_n_nodes"] = r.n_nodes
                row[f"{label}_qe"] = round(r.qe, 4)

        rows.append(row)
        parts = [
            f"DBGSOM={r_dbgsom.time:.2f}s",
            f"DBGSOM_textbook={r_dbgsom_tb.time:.2f}s",
            f"n={r_dbgsom.n_nodes}",
        ]
        competitors = [
            ("MiniSom", r_minisom),
            ("SuSi", r_susi),
            ("torchsom_cpu", r_torchsom_cpu),
            ("torchsom_gpu", r_torchsom_gpu),
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

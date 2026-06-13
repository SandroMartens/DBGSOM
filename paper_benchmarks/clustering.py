"""Paper benchmark: clustering comparison on Digits dataset.
DBGSOM vs MiniSom vs SuSi — reproducible, fixed seed.

Usage:
    uv run python paper_benchmarks/clustering.py

Output:
    paper_benchmarks/results/clustering_digits.csv
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dbgsom import SomVQ

RANDOM_STATE = 42
RESULTS_DIR = Path(__file__).parent / "results"

DBGSOM_PARAMS = dict(
    n_iter=500,
    lambda_=15.8,
    sigma_end=1,
    # max_neurons=100,
    random_state=RANDOM_STATE,
)


def load_data():
    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data)
    y = digits.target
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def _row(name, n_proto, elapsed, qe, te, y_true, labels):
    return {
        "Algorithm": name,
        "n_prototypes": n_proto,
        "Time (s)": round(elapsed, 3),
        "QE": round(qe, 4),
        "TE": round(te, 4) if te is not None else "—",
        "ARI": round(adjusted_rand_score(y_true, labels), 3),
        "Silhouette": round(silhouette_score(X_test_global, labels), 3),
    }


X_test_global: np.ndarray  # set in main


def _topographic_error(X: np.ndarray, weights: np.ndarray, side: int) -> float:
    """Fraction of samples where 1st and 2nd BMU are not grid-adjacent."""
    dists = np.linalg.norm(X[:, None, :] - weights[None, :, :], axis=2)
    top2 = np.argpartition(dists, kth=1, axis=1)[:, :2]
    r1, c1 = top2[:, 0] // side, top2[:, 0] % side
    r2, c2 = top2[:, 1] // side, top2[:, 1] % side
    return float(np.mean((np.abs(r1 - r2) + np.abs(c1 - c2)) != 1))


def train_dbgsom(X_train, X_test, y_test):
    t0 = time.perf_counter()
    som = SomVQ(**DBGSOM_PARAMS)
    som.fit(X_train)
    elapsed = time.perf_counter() - t0
    labels = som.predict(X_test)
    qe = float(som.calculate_quantization_error(X_test))
    te = float(som.topographic_error_)
    n = len(som.neurons_)
    print(f"  DBGSOM: {n} neurons, {elapsed:.3f}s")
    return _row("DBGSOM", n, elapsed, qe, te, y_test, labels), n


def train_minisom(X_train, X_test, y_test, n_neurons):
    try:
        from minisom import MiniSom
    except ImportError:
        print("  MiniSom not installed — skip (pip install minisom)")
        return None

    side = math.ceil(math.sqrt(n_neurons))
    t0 = time.perf_counter()
    som = MiniSom(
        side,
        side,
        X_train.shape[1],
        sigma=0.2 * np.sqrt(n_neurons),
        learning_rate=0.5,
        random_seed=RANDOM_STATE,
    )
    som.train(X_train, num_iteration=500 * n_neurons, verbose=False)
    elapsed = time.perf_counter() - t0

    def _winner_flat(x):
        r, c = som.winner(x)
        return r * side + c

    labels = np.array([_winner_flat(x) for x in X_test])
    weights_flat = som.get_weights().reshape(-1, X_train.shape[1])
    qe = float(
        np.mean([np.linalg.norm(x - weights_flat[_winner_flat(x)]) for x in X_test])
    )
    te = _topographic_error(X_test, weights_flat, side)
    print(f"  MiniSom ({side}×{side}): {elapsed:.3f}s")
    return _row("MiniSom", side * side, elapsed, qe, te, y_test, labels)


def train_susi(X_train, X_test, y_test, n_neurons):
    try:
        from susi import SOMClustering
    except ImportError:
        print("  SuSi not installed — skip (pip install susi)")
        return None

    side = math.ceil(math.sqrt(n_neurons))
    t0 = time.perf_counter()
    som = SOMClustering(n_rows=side, n_columns=side, random_state=RANDOM_STATE)
    som.fit(X_train)
    elapsed = time.perf_counter() - t0

    bmus = np.array(som.get_bmus(X_test))  # (n, 2)
    rows_idx = bmus[:, 0]
    cols_idx = bmus[:, 1]
    labels = rows_idx * side + cols_idx
    bmu_weights = som.unsuper_som_[rows_idx, cols_idx]
    qe = float(np.mean(np.linalg.norm(X_test - bmu_weights, axis=1)))
    weights_flat = som.unsuper_som_.reshape(-1, X_test.shape[1])
    te = _topographic_error(X_test, weights_flat, side)
    print(f"  SuSi ({side}×{side}): {elapsed:.3f}s")
    return _row("SuSi", side * side, elapsed, qe, te, y_test, labels)


def train_torchsom(X_train, X_test, y_test, n_neurons):
    try:
        import torch
        from torchsom import SOM
    except ImportError:
        print("  torchsom not installed — skip (pip install torchsom)")
        return None

    side = math.ceil(math.sqrt(n_neurons))
    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    t0 = time.perf_counter()
    som = SOM(
        side,
        side,
        X_train.shape[1],
        epochs=36,
        sigma=0.2 * np.sqrt(n_neurons),
        batch_size=len(X_train),  # full-batch
    )
    som.fit(X_train_t)
    elapsed = time.perf_counter() - t0

    # BMU + QE from weights directly — avoids depending on forward pass API
    weights = som.weights.detach().cpu().numpy().reshape(-1, X_train.shape[1])
    dists = np.linalg.norm(X_test[:, None, :] - weights[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    qe = float(np.mean(np.min(dists, axis=1)))
    te = _topographic_error(X_test, weights, side)

    print(f"  torchsom ({side}×{side}): {elapsed:.3f}s")
    return _row("torchsom", side * side, elapsed, qe, te, y_test, labels)


def main():
    global X_test_global
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()
    X_test_global = X_test

    print("Digits benchmark (random_state=42)")
    row_dbgsom, n_neurons = train_dbgsom(X_train, X_test, y_test)
    row_minisom = train_minisom(X_train, X_test, y_test, n_neurons)
    row_susi = train_susi(X_train, X_test, y_test, n_neurons)
    row_torchsom = train_torchsom(X_train, X_test, y_test, n_neurons)

    all_rows = [row_dbgsom, row_minisom, row_susi, row_torchsom]
    rows = [r for r in all_rows if r is not None]
    df = pd.DataFrame(rows).set_index("Algorithm")

    out = RESULTS_DIR / "clustering_digits.csv"
    df.to_csv(out)
    print(f"\nSaved → {out}")
    print(df.to_string())


if __name__ == "__main__":
    main()

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
    max_neurons=100,
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
        sigma=1.0,
        learning_rate=0.5,
        random_seed=RANDOM_STATE,
    )
    som.train(X_train, num_iteration=len(X_train) * 500, verbose=False)
    elapsed = time.perf_counter() - t0

    def _winner_flat(x):
        r, c = som.winner(x)
        return r * side + c

    labels = np.array([_winner_flat(x) for x in X_test])
    weights = som.get_weights()
    qe = float(np.mean([np.linalg.norm(x - weights[som.winner(x)]) for x in X_test]))
    print(f"  MiniSom ({side}×{side}): {elapsed:.3f}s")
    return _row("MiniSom", side * side, elapsed, qe, None, y_test, labels)


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
    print(f"  SuSi ({side}×{side}): {elapsed:.3f}s")
    return _row("SuSi", side * side, elapsed, qe, None, y_test, labels)


def main():
    global X_test_global
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()
    X_test_global = X_test

    print("Digits benchmark (random_state=42)")
    row_dbgsom, n_neurons = train_dbgsom(X_train, X_test, y_test)
    row_minisom = train_minisom(X_train, X_test, y_test, n_neurons)
    row_susi = train_susi(X_train, X_test, y_test, n_neurons)

    rows = [r for r in [row_dbgsom, row_minisom, row_susi] if r is not None]
    df = pd.DataFrame(rows).set_index("Algorithm")

    out = RESULTS_DIR / "clustering_digits.csv"
    df.to_csv(out)
    print(f"\nSaved → {out}")
    print(df.to_string())


if __name__ == "__main__":
    main()

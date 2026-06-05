"""Quality + runtime comparison: pointer_search modes and radii."""

import time
from statistics import mean

import numpy as np
from sklearn.datasets import load_digits, make_blobs

from dbgsom.SomClassifier import SomClassifier
from dbgsom.SomVQ import SomVQ

REPS = 3


def run_vq(X, mode, radius, seed, **kwargs):
    som = SomVQ(
        random_state=seed, pointer_search=mode, pointer_search_radius=radius, **kwargs
    )
    t0 = time.perf_counter()
    som.fit(X)
    elapsed = time.perf_counter() - t0
    return (
        elapsed,
        som.quantization_error_,
        som.topographic_error_,
        som.som_.number_of_nodes(),
    )


def run_clf(X, y, mode, radius, seed, **kwargs):
    clf = SomClassifier(
        random_state=seed, pointer_search=mode, pointer_search_radius=radius, **kwargs
    )
    clf.fit(X, y)
    return clf.score(X, y)


def report(label, X, y=None, configs=None, **kwargs):
    print(f"\n=== {label} ===")
    hdr = f"{'config':14} | {'t (s)':>7} | {'QE':>7} | {'TE':>7} | {'nodes':>6}"
    if y is not None:
        hdr += f" | {'score':>7}"
    print(hdr)
    print("-" * (len(hdr) + 2))

    for mode, radius in configs:
        tag = f"{mode}/r={radius}" if mode != "none" else "none"
        times, qes, tes, nodes, scores = [], [], [], [], []
        for seed in range(REPS):
            t, qe, te, n = run_vq(X, mode, radius, seed, **kwargs)
            times.append(t)
            qes.append(qe)
            tes.append(te)
            nodes.append(n)
            if y is not None:
                scores.append(run_clf(X, y, mode, radius, seed, **kwargs))

        line = (
            f"{tag:14} | {mean(times):>7.2f} | {mean(qes):>7.3f} |"
            f" {mean(tes):>7.3f} | {mean(nodes):>6.1f}"
        )
        if y is not None:
            line += f" | {mean(scores):>7.3f}"
        print(line)


CONFIGS = [
    ("none", 1),
    ("fine", 1),
    ("fine", 2),
    ("fine", 3),
    ("all", 1),
    ("all", 2),
    ("all", 3),
]

X_d, y_d = load_digits(return_X_y=True)
report("digits (N=1797, D=64)", X_d, y_d, configs=CONFIGS, n_iter=500, max_neurons=30)

X_s, _ = make_blobs(n_samples=50_000, n_features=20, centers=10, random_state=0)
X_s = X_s.astype(np.float64)
report("synthetic (N=50k, D=20)", X_s, configs=CONFIGS, n_iter=200, max_neurons=100)

"""Paper visualization: DBGSOM grid and PCA projection on Digits dataset.

Saves two figures with majority-class coloring per neuron:
  - som_grid.png   — SOM topology on the 2D grid
  - som_pca.png    — neuron weights projected to PCA space

Usage:
    uv run python paper_benchmarks/visualization.py

Output:
    paper_benchmarks/results/som_grid.png
    paper_benchmarks/results/som_pca.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from dbgsom import SomVQ

RANDOM_STATE = 42
RESULTS_DIR = Path(__file__).parent / "results"
DPI = 150

DBGSOM_PARAMS = dict(
    n_iter=500,
    lambda_=15.8,
    max_neurons=100,
    sigma_start=2,
    sigma_end=1,
    random_state=RANDOM_STATE,
)

DIGIT_COLORS = plt.get_cmap("tab10").colors  # 10 distinct colors


def majority_class(labels: np.ndarray, n_neurons: int, n_classes: int) -> np.ndarray:
    """For each neuron, return the most common class among assigned samples."""
    counts = np.zeros((n_neurons, n_classes), dtype=int)
    for idx, cls in zip(labels, y_global):
        counts[idx, cls] += 1
    winner = counts.argmax(axis=1)
    # neurons with no hits get -1
    winner[counts.sum(axis=1) == 0] = -1
    return winner


y_global: np.ndarray  # set in main


def _draw(ax, positions, edges, node_classes, hit_counts, title):
    """Draw SOM on a matplotlib Axes."""
    for u, v in edges:
        xu, yu = positions[u]
        xv, yv = positions[v]
        ax.plot([xu, xv], [yu, yv], color="#cccccc", linewidth=0.8, zorder=1)

    for idx, (x, y) in enumerate(positions):
        cls = node_classes[idx]
        color = DIGIT_COLORS[cls] if cls >= 0 else "#888888"
        size = 20 + 60 * (hit_counts[idx] / max(hit_counts.max(), 1))
        ax.scatter(
            x, y, s=size, color=color, zorder=2, edgecolors="white", linewidths=0.4
        )

    # legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=DIGIT_COLORS[i],
            markersize=7,
            label=str(i),
        )
        for i in range(10)
    ]
    ax.legend(handles=handles, title="Digit", loc="best", fontsize=7, title_fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    global y_global
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data)
    y = digits.target
    y_global = y

    print("Training DBGSOM on Digits...")
    som = SomVQ(**DBGSOM_PARAMS)
    som.fit(X)
    n = len(som.neurons_)
    print(f"  {n} neurons")

    labels = som.predict(X)  # flat neuron index per sample
    hit_counts = np.bincount(labels, minlength=n).astype(float)
    node_classes = majority_class(labels, n, 10)

    edges = [
        (som._node_to_idx[u], som._node_to_idx[v])
        for u, v in som.som_.edges()
        if u in som._node_to_idx and v in som._node_to_idx
    ]

    # --- Grid layout ---
    grid_pos = np.array([som.neurons_[i] for i in range(n)], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    _draw(
        ax, grid_pos, edges, node_classes, hit_counts, "DBGSOM — Grid layout (Digits)"
    )
    out_grid = RESULTS_DIR / "som_grid.png"
    fig.savefig(out_grid, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_grid}")

    # --- PCA layout ---
    from sklearn.decomposition import PCA

    weights = som.weights_
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_pos = pca.fit(X).transform(weights)

    fig, ax = plt.subplots(figsize=(6, 6))
    _draw(ax, pca_pos, edges, node_classes, hit_counts, "DBGSOM — PCA layout (Digits)")
    out_pca = RESULTS_DIR / "som_pca.png"
    fig.savefig(out_pca, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_pca}")


if __name__ == "__main__":
    main()

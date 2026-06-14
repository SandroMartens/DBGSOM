"""Plot scaling benchmark results: training time vs. dataset size N.

Usage:
    uv run python paper_benchmarks/plot_scaling.py

Input:
    paper_benchmarks/results/scaling.csv

Output:
    paper_benchmarks/results/scaling.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"

SERIES = [
    ("DBGSOM_time", "DBGSOM (fast path)", "#1f77b4", "-", "o"),
    ("DBGSOM_textbook_time", "DBGSOM (textbook)", "#1f77b4", "--", "s"),
    ("MiniSom_time", "MiniSom", "#ff7f0e", "-", "^"),
    ("SuSi_time", "SuSi", "#2ca02c", "-", "D"),
    ("torchsom_cpu_time", "torchsom (CPU)", "#9467bd", "-", "v"),
    ("torchsom_gpu_time", "torchsom (GPU)", "#d62728", "-", "P"),
]


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / "scaling.csv", index_col="N")

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
    ax.set_facecolor("white")

    for col, label, color, ls, marker in SERIES:
        if col not in df.columns:
            continue
        data = df[col].dropna()
        ax.plot(
            data.index,
            data.values,
            label=label,
            color=color,
            linestyle=ls,
            marker=marker,
            markersize=5,
            linewidth=1.5,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset size N", fontsize=11)
    ax.set_ylabel("Training time (s)", fontsize=11)
    ax.set_title("Training time vs. dataset size", fontsize=12)
    ax.legend(fontsize=9, framealpha=1.0, edgecolor="#cccccc")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, color="#dddddd")
    ax.spines[["top", "right"]].set_visible(False)

    out = RESULTS_DIR / "scaling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

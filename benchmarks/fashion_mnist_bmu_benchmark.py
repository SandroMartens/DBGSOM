"""Real training benchmark: Numba vs BLAS Euclidean BMU on Fashion-MNIST.

Patches _get_winning_neurons to swap the n_bmu=1 Euclidean path between
the current BLAS implementation and the reconstructed Numba kernel
(removed in commit ed599b0). Measures full fit() time.
"""

import time
import types

import numba as nb
import numpy as np
import numpy.typing as npt
from sklearn.datasets import fetch_openml
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler, normalize

from dbgsom.BaseSom import (
    numba_find_winners_cosine,
    numba_find_winners_pointer,
)
from dbgsom.SomVQ import SomVQ

REPS = 3
N_ITER = 30
MAX_NEURONS = 50


# ── reconstructed Numba kernel ────────────────────────────────────────────────


@nb.njit(cache=True, parallel=True, fastmath=True)
def numba_find_winners_euclidean(
    data: npt.NDArray, weights: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """Reconstructed from commit ed599b0 (removed in that commit)."""
    n_samples = data.shape[0]
    n_features = data.shape[1]
    n_neurons = weights.shape[0]
    winners = np.empty(n_samples, dtype=np.int64)
    distances = np.empty(n_samples, dtype=data.dtype)
    for i in nb.prange(n_samples):
        best_dist_sq = np.inf
        best_j = 0
        for j in range(n_neurons):
            d_sq = 0.0
            for k in range(n_features):
                diff = data[i, k] - weights[j, k]
                d_sq += diff * diff
            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                best_j = j
        winners[i] = best_j
        distances[i] = np.sqrt(best_dist_sq)
    return distances, winners


# ── patched _get_winning_neurons using Numba for Euclidean n_bmu=1 ──────────


def _get_winning_neurons_numba(self, data: npt.NDArray, n_bmu: int, prev_winners=None):
    if (
        prev_winners is not None
        and n_bmu == 1
        and self.metric != "cosine"
        and (
            self.pointer_search == "all"
            or (self.pointer_search == "fine" and self._training_phase == "fine")
        )
    ):
        return numba_find_winners_pointer(
            data, self.weights_, prev_winners, self._neighbor_matrix
        )
    if self.metric == "cosine":
        data = normalize(data)
        if n_bmu == 1:
            return numba_find_winners_cosine(data, self.weights_)
        sim_matrix = data @ self.weights_.T
        dist_matrix = 1.0 - sim_matrix
        part = np.argpartition(dist_matrix, n_bmu, axis=1)[:, :n_bmu]
        row_idx = np.arange(len(data))[:, np.newaxis]
        order = np.argsort(dist_matrix[row_idx, part], axis=1)
        winners = part[row_idx, order]
        distances = dist_matrix[row_idx, winners]
        return distances, winners
    # ── Numba path for Euclidean ──
    if n_bmu == 1:
        return numba_find_winners_euclidean(data, self.weights_)
    dist_matrix = euclidean_distances(data, self.weights_)
    part = np.argpartition(dist_matrix, n_bmu, axis=1)[:, :n_bmu]
    row_idx = np.arange(len(data))[:, np.newaxis]
    order = np.argsort(dist_matrix[row_idx, part], axis=1)
    winners = part[row_idx, order]
    distances = dist_matrix[row_idx, winners]
    return distances, winners


def train(X, use_numba: bool, seed: int) -> float:
    som = SomVQ(
        n_iter=N_ITER, max_neurons=MAX_NEURONS, random_state=seed, verbose=False
    )
    if use_numba:
        som._get_winning_neurons = types.MethodType(_get_winning_neurons_numba, som)
    t0 = time.perf_counter()
    som.fit(X)
    return time.perf_counter() - t0


def main():
    print("Loading Fashion-MNIST ...", flush=True)
    fmnist = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto")
    X_full = StandardScaler().fit_transform(fmnist.data.astype(np.float64))
    X = X_full[:10_000]
    print(f"X shape: {X.shape}  (n=10k, d=784)\n")

    # JIT warmup on tiny data
    Xw = np.random.randn(8, 784)
    Ww = np.random.randn(8, 784)
    numba_find_winners_euclidean(Xw, Ww)

    blas_times, numba_times = [], []
    for rep in range(REPS):
        t_blas = train(X, use_numba=False, seed=rep)
        t_numba = train(X, use_numba=True, seed=rep)
        blas_times.append(t_blas)
        numba_times.append(t_numba)
        print(
            f"rep {rep + 1}: BLAS={t_blas:.1f}s  Numba={t_numba:.1f}s  ratio={t_blas / t_numba:.2f}x"
        )

    print()
    print(f"mean BLAS : {np.mean(blas_times):.1f}s")
    print(f"mean Numba: {np.mean(numba_times):.1f}s")
    print(f"Numba speedup over BLAS: {np.mean(blas_times) / np.mean(numba_times):.2f}x")


if __name__ == "__main__":
    main()

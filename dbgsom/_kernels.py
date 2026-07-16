"""Numba JIT kernels and decay functions used by BaseSom."""

from math import exp
from typing import Callable

import numba as nb
import numpy as np
import numpy.typing as npt


def linear_decay(
    sigma_start: float,
    sigma_end: float,
    max_iter: int,
    current_iter: float,
    learning_rate: None = None,
) -> float:
    """Linear decay between sigma_start and sigma_end over t training iterations."""
    ratio = current_iter / max_iter
    sigma = sigma_start * (1 - ratio) + sigma_end * ratio

    return sigma


def exponential_decay(
    sigma_start: float,
    sigma_end: float,
    max_iter: int,
    current_iter: float,
    learning_rate: float,
) -> float:
    """Exponential decay between sigma_start and sigma_end with a given learning rate."""
    sigma = sigma_end + (sigma_start - sigma_end) * exp(-learning_rate * current_iter)

    return sigma


_DECAY_FUNCTIONS: dict[str, Callable[..., float]] = {
    "linear": linear_decay,
    "exponential": exponential_decay,
}


@nb.njit(cache=True, parallel=True, fastmath=True)
def numba_voronoi_set_centers(
    kernel,
    data: npt.NDArray,
    shape: tuple,
    groups: npt.NDArray,
    offsets: npt.NDArray,
    index: npt.NDArray,
) -> np.ndarray:
    """Calculate the centers of the Voronoi regions based on the winners and data arrays."""
    voronoi_set_centers = np.zeros(shape=shape, dtype=data.dtype)
    for i in nb.prange(groups.size):  # ty:ignore[not-iterable]
        group_start = offsets[i]
        group_end = offsets[i + 1] if i + 1 < groups.size else index.size
        group_index = index[group_start:group_end]
        weight_sum = np.sum(kernel[group_index])
        n_s = group_index.shape[0]
        n_f = shape[1]
        neuron_idx = groups[i]
        if weight_sum == 0.0:
            # All kernel weights underflowed: fall back to unweighted mean.
            for s in range(n_s):
                row = data[group_index[s]]
                for j in range(n_f):
                    voronoi_set_centers[neuron_idx, j] += row[j]
            for j in range(n_f):
                voronoi_set_centers[neuron_idx, j] /= n_s
        else:
            for s in range(n_s):
                w = kernel[group_index[s]]
                row = data[group_index[s]]
                for j in range(n_f):
                    voronoi_set_centers[neuron_idx, j] += w * row[j]
            for j in range(n_f):
                voronoi_set_centers[neuron_idx, j] /= weight_sum

    return voronoi_set_centers


@nb.njit(cache=True, parallel=True, fastmath=True)
def numba_find_winners_euclidean(
    data: npt.NDArray, weights: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """Find the nearest weight vector per sample (fused distance + argmin).

    Chosen over BLAS euclidean_distances: ~2x faster on AMD/OpenBLAS;
    Intel/MKL closes the gap but the AMD penalty of BLAS is larger than
    the Intel penalty of Numba, making Numba the better library default.
    """
    n_samples = data.shape[0]
    n_features = data.shape[1]
    n_neurons = weights.shape[0]
    winners = np.empty(n_samples, dtype=np.int64)
    distances = np.empty(n_samples, dtype=data.dtype)
    for i in nb.prange(n_samples):  # ty:ignore[not-iterable]
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


@nb.njit(cache=True, parallel=True, fastmath=True)
def numba_find_winners_cosine(
    data: npt.NDArray, weights: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """Find the most similar weight vector per sample via fused dot-product + argmax.

    Assumes data and weights are already L2-normalised (unit vectors).
    No n×m similarity matrix is allocated.
    """
    n_samples = data.shape[0]
    n_features = data.shape[1]
    n_neurons = weights.shape[0]
    winners = np.empty(n_samples, dtype=np.int64)
    distances = np.empty(n_samples, dtype=data.dtype)
    for i in nb.prange(n_samples):  # ty:ignore[not-iterable]
        best_sim = -np.inf
        best_j = 0
        for j in range(n_neurons):
            sim = 0.0
            for k in range(n_features):
                sim += data[i, k] * weights[j, k]
            if sim > best_sim:
                best_sim = sim
                best_j = j
        winners[i] = best_j
        distances[i] = 1.0 - best_sim
    return distances, winners


@nb.njit(cache=True, parallel=True, fastmath=True)
def numba_find_winners_pointer(
    data: npt.NDArray,
    weights: npt.NDArray,
    prev_winners: npt.NDArray,
    neighbor_matrix: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Find BMU per sample by searching only prev_winner + its graph neighbors."""
    n = data.shape[0]
    winners = np.empty(n, np.int64)
    distances = np.empty(n, np.float64)
    for i in nb.prange(n):  # ty:ignore[not-iterable]
        pw = prev_winners[i]
        best_d = np.sum((data[i] - weights[pw]) ** 2)
        best_idx = pw
        for j in range(neighbor_matrix.shape[1]):
            nidx = neighbor_matrix[pw, j]
            if nidx < 0:
                break
            d = np.sum((data[i] - weights[nidx]) ** 2)
            if d < best_d:
                best_d = d
                best_idx = nidx
        winners[i] = best_idx
        distances[i] = np.sqrt(best_d)
    return distances, winners


@nb.njit(cache=True, parallel=True, fastmath=True)
def numba_find_winners_pointer_cosine(
    data: npt.NDArray,
    weights: npt.NDArray,
    prev_winners: npt.NDArray,
    neighbor_matrix: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Find BMU per sample via pointer search using cosine similarity.

    Searches only prev_winner + its graph neighbors.
    Assumes data and weights are already L2-normalised (unit vectors).
    """
    n = data.shape[0]
    n_features = data.shape[1]
    winners = np.empty(n, np.int64)
    distances = np.empty(n, np.float64)
    for i in nb.prange(n):  # ty:ignore[not-iterable]
        pw = prev_winners[i]
        sim = 0.0
        for k in range(n_features):
            sim += data[i, k] * weights[pw, k]
        best_sim = sim
        best_idx = pw
        for j in range(neighbor_matrix.shape[1]):
            nidx = neighbor_matrix[pw, j]
            if nidx < 0:
                break
            sim = 0.0
            for k in range(n_features):
                sim += data[i, k] * weights[nidx, k]
            if sim > best_sim:
                best_sim = sim
                best_idx = nidx
        winners[i] = best_idx
        distances[i] = 1.0 - best_sim
    return distances, winners

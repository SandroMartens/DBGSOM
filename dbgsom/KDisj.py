"""KDisj: SOM for qualitative data in disjunctive table form.

Reference: Cottrell et al., "Analyzing a contingency table with Kohonen maps:
a Factorial Correspondence Analysis", ICANN, 1993.

Basis algorithm from: G. Cabanes et al., Neural Networks 32 (2012) 186-195.

Experimental: not part of the stable public API (SomVQ/SomClassifier).
No backwards-compatibility guarantees.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.utils import check_random_state


class KDisj:
    """SOM for qualitative data encoded as a disjunctive (binary) table.

    Maps both data points (rows) and modalities (columns) of a binary
    table T into a shared (A + E)-dimensional prototype space, where
    A = number of modalities and E = number of data points.

    Parameters
    ----------
    n_rows : int
        Number of rows in the SOM grid.
    n_cols : int
        Number of columns in the SOM grid.
    n_iter : int, default=200
        Number of training epochs.
    sigma_start : float or None, default=None
        Initial neighborhood radius. Defaults to max(n_rows, n_cols) / 2.
    sigma_end : float, default=0.5
        Final neighborhood radius.
    learning_rate_start : float, default=0.5
        Initial learning rate.
    learning_rate_end : float, default=0.01
        Final learning rate.
    random_state : int or None, default=None
        Random seed for reproducibility.

    """

    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        n_iter: int = 200,
        sigma_start: float | None = None,
        sigma_end: float = 0.5,
        learning_rate_start: float = 0.5,
        learning_rate_end: float = 0.01,
        random_state: int | None = None,
    ) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_iter = n_iter
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.learning_rate_start = learning_rate_start
        self.learning_rate_end = learning_rate_end
        self.random_state = random_state

    def fit(self, T: npt.ArrayLike) -> KDisj:
        """Train KDisj on a disjunctive table.

        Parameters
        ----------
        T : array-like of shape (E, A)
            Binary disjunctive table. Each row is a data point; each column
            is a modality. T[i, j] = 1 iff data point i has modality j.

        Returns
        -------
        self

        """
        T = np.asarray(T, dtype=np.float64)
        if T.ndim != 2:
            raise ValueError("T must be a 2D array.")
        if not np.all((T == 0) | (T == 1)):
            raise ValueError("T must be binary (0/1).")

        E, A = T.shape
        self.E_ = E
        self.A_ = A
        K = self.n_rows * self.n_cols
        rng = check_random_state(self.random_state)
        sigma_start = self.sigma_start or max(self.n_rows, self.n_cols) / 2.0

        self.grid_distances_ = self._precompute_grid_distances()

        # Prototypes shape (K, A+E): init from random rows/columns of T
        self.weights_ = self._init_prototypes(T, K, rng)

        # Rarest modality per data point: argmin column_sum where T[i,j]==1
        column_sums = T.sum(axis=0)
        rarest = np.array([self._rarest_modality(T[i], column_sums) for i in range(E)])

        for epoch in range(self.n_iter):
            t = epoch / max(self.n_iter - 1, 1)
            lr = (
                self.learning_rate_start
                * (self.learning_rate_end / self.learning_rate_start) ** t
            )
            sigma = sigma_start * (self.sigma_end / sigma_start) ** t

            order = rng.permutation(E)
            for i in order:
                j = rarest[i]

                # Data presentation: BMU on first A dims
                x_full = np.concatenate([T[i], T[:, j]])
                bmu_d = self._find_bmu(x_full[:A], self.weights_[:, :A])
                h = self._neighborhood(bmu_d, sigma)
                self.weights_ += lr * h[:, None] * (x_full - self.weights_)

                # Modality presentation: BMU on last E dims, update only last E dims
                bmu_m = self._find_bmu(T[:, j], self.weights_[:, A:])
                h_m = self._neighborhood(bmu_m, sigma)
                self.weights_[:, A:] += (
                    lr * h_m[:, None] * (T[:, j] - self.weights_[:, A:])
                )

        return self

    def transform(self, T: npt.ArrayLike) -> np.ndarray:
        """Find the best matching unit for each data point.

        Uses only the first A dimensions (modality space) for BMU search,
        so T does not need to be the same table used for training.

        Parameters
        ----------
        T : array-like of shape (n_samples, A)
            Binary data rows. Each row must have the same number of modalities
            as the training data.

        Returns
        -------
        bmu_indices : np.ndarray of shape (n_samples,)
            Flat neuron index (row * n_cols + col) of the BMU for each sample.

        """
        T = np.asarray(T, dtype=np.float64)
        A = self.A_
        return np.array(
            [self._find_bmu(T[i], self.weights_[:, :A]) for i in range(len(T))]
        )

    def plot(self, labels: npt.ArrayLike | None = None) -> None:
        """Visualize neurons projected to 2D via PCA of the first A dimensions.

        Parameters
        ----------
        labels : array-like of shape (n_samples,), optional
            Sample labels. When provided, scatter-plots samples colored by label.

        """
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        K = self.n_rows * self.n_cols
        A = self.A_
        W_data = self.weights_[:, :A]

        n_components = min(2, A, K)
        if n_components < 2:
            raise ValueError("Need at least 2 modalities to plot.")

        coords = PCA(n_components=2).fit_transform(W_data)
        xs, ys = coords[:, 0], coords[:, 1]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(xs, ys, c="steelblue", s=60, zorder=3, label="neurons")

        # Annotate grid positions
        for k in range(K):
            row, col = divmod(k, self.n_cols)
            ax.annotate(
                f"({row},{col})", (xs[k], ys[k]), fontsize=6, ha="center", va="bottom"
            )

        if labels is not None:
            bmus = self.transform(
                np.asarray(self._last_T_)
                if hasattr(self, "_last_T_")
                else np.zeros((len(labels), A))
            )
            scatter_x = xs[bmus]
            scatter_y = ys[bmus]
            labels = np.asarray(labels)
            for lbl in np.unique(labels):
                mask = labels == lbl
                ax.scatter(
                    scatter_x[mask],
                    scatter_y[mask],
                    s=20,
                    alpha=0.5,
                    label=str(lbl),
                    zorder=2,
                )

        ax.set_title("KDisj — neuron positions (PCA of modality dims)")
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _precompute_grid_distances(self) -> np.ndarray:
        """Manhattan distance between all neuron pairs on the grid."""
        K = self.n_rows * self.n_cols
        coords = np.array(
            [[r, c] for r in range(self.n_rows) for c in range(self.n_cols)]
        )
        diffs = np.abs(coords[:, None, :] - coords[None, :, :])
        return diffs.sum(axis=-1).astype(np.float64)  # (K, K)

    def _init_prototypes(
        self, T: np.ndarray, K: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Initialize prototypes from random rows and columns of T."""
        E, A = T.shape
        W = np.zeros((K, A + E))
        row_idx = rng.choice(E, size=K, replace=True)
        col_idx = rng.choice(A, size=K, replace=True)
        for k in range(K):
            W[k, :A] = T[row_idx[k]]
            W[k, A:] = T[:, col_idx[k]]
        return W

    @staticmethod
    def _rarest_modality(row: np.ndarray, column_sums: np.ndarray) -> int:
        """Index of the rarest non-null modality in a data row."""
        active = np.where(row > 0)[0]
        if len(active) == 0:
            return 0
        return int(active[np.argmin(column_sums[active])])

    def _find_bmu(self, vector: np.ndarray, weight_slice: np.ndarray) -> int:
        """Return index of prototype closest to vector (Euclidean)."""
        diffs = weight_slice - vector
        dists = np.einsum("ki,ki->k", diffs, diffs)
        return int(np.argmin(dists))

    def _neighborhood(self, bmu: int, sigma: float) -> np.ndarray:
        """Gaussian neighborhood centered on bmu."""
        d2 = self.grid_distances_[bmu] ** 2
        return np.exp(-d2 / (2 * sigma**2))

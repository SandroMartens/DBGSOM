"""Batch LVQ-SOM (Kohonen 2001, sec. 6.11): supervised refinement of a
fitted SomVQ's prototypes via a class-sign-flipped neighborhood kernel.

See docs/superpowers/specs/2026-07-15-batch-lvq-som-design.md for the
full derivation.
"""

from numbers import Integral, Real

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from ._kernels import numba_find_winners_euclidean


class BatchLvqSom(ClassifierMixin, BaseEstimator):
    """Refine a fitted SomVQ's prototypes with the batch LVQ-SOM rule.

    Parameters
    ----------
    n_iter : int, default=10
        Number of refinement iterations.
    sigma : float or None, default=None
        Fixed neighborhood bandwidth. If None, resolved at fit time from
        ``som._calculate_current_sigma()``.

    Attributes
    ----------
    weights_ : ndarray of shape (n_prototypes, n_features)
        Refined prototype weight vectors.
    node_labels_ : ndarray of shape (n_prototypes,)
        Majority-vote class label of each prototype after the final
        iteration.
    classes_ : ndarray of shape (n_classes,)
        Unique class labels seen during fit.

    """

    _parameter_constraints = {
        "n_iter": [Interval(Integral, 1, None, closed="left")],
        "sigma": [Interval(Real, 0, None, closed="neither"), None],
    }

    def __init__(self, n_iter: int = 10, sigma: float | None = None) -> None:
        """Initialize BatchLvqSom.

        Parameters
        ----------
        n_iter : int, default=10
            Number of refinement iterations.
        sigma : float or None, default=None
            Fixed neighborhood bandwidth.

        """
        self.n_iter = n_iter
        self.sigma = sigma

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike, som) -> "BatchLvqSom":  # noqa: ANN001 -- som is duck-typed (SomVQ-like); no Protocol precedent in this codebase
        """Refine ``som``'s prototypes with labelled data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Class labels.
        som : object
            Already-fitted instance. Supplies initial weights, graph
            topology, and neighborhood bandwidth. Not mutated.

        Returns
        -------
        self : BatchLvqSom

        """
        self._validate_params()
        try:
            check_is_fitted(som, attributes=["weights_"])
        except TypeError:
            if not hasattr(som, "weights_"):
                raise NotFittedError(
                    "som does not have weights_ attribute. "
                    "Make sure to fit the SOM before passing it here."
                ) from None
        X, y = validate_data(self, X, y, dtype="numeric")

        weights = som.weights_.copy()
        n_neurons, n_features = weights.shape
        if X.shape[1] != n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but som.weights_ has {n_features}."
            )

        check_classification_targets(y)
        classes, y_idx = np.unique(y, return_inverse=True)
        self.classes_ = classes
        n_classes = len(classes)

        sigma = self.sigma if self.sigma is not None else som._calculate_current_sigma()
        dm = som._distance_matrix.astype(np.float64)
        neighborhood = np.exp(-(dm**2) / (2 * sigma**2))

        row_idx = np.arange(n_neurons)
        node_labels = np.zeros(n_neurons, dtype=np.int64)
        for _ in range(self.n_iter):
            _, winners = numba_find_winners_euclidean(X, weights)

            flat_idx = winners * n_classes + y_idx
            n_bins = n_neurons * n_classes
            class_sums = np.zeros((n_bins, n_features))
            np.add.at(class_sums, flat_idx, X)
            class_sums = class_sums.reshape(n_neurons, n_classes, n_features)
            class_counts = (
                np.bincount(flat_idx, minlength=n_bins)
                .astype(np.float64)
                .reshape(n_neurons, n_classes)
            )

            node_labels = np.argmax(class_counts, axis=1)

            weighted_sums = (
                neighborhood @ class_sums.reshape(n_neurons, n_classes * n_features)
            ).reshape(n_neurons, n_classes, n_features)
            weighted_counts = neighborhood @ class_counts

            numerator = 2 * weighted_sums[row_idx, node_labels, :] - weighted_sums.sum(
                axis=1
            )
            denominator = 2 * weighted_counts[
                row_idx, node_labels
            ] - weighted_counts.sum(axis=1)

            update_mask = denominator > 0
            weights[update_mask] = (
                numerator[update_mask] / denominator[update_mask, None]
            )

        self.weights_ = weights
        self.node_labels_ = classes[node_labels]
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray:
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted class labels.

        """
        check_is_fitted(self, attributes=["weights_"])
        X = np.asarray(validate_data(self, X, reset=False))
        _, winners = numba_find_winners_euclidean(X, self.weights_)
        return self.node_labels_[winners]

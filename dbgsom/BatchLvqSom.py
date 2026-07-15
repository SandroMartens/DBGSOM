"""Batch LVQ-SOM (Kohonen 2001, sec. 6.11): supervised refinement of a
fitted SomVQ's prototypes via a class-sign-flipped neighborhood kernel.

See docs/superpowers/specs/2026-07-15-batch-lvq-som-design.md for the
full derivation.
"""

from numbers import Integral, Real

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import Interval
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

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> "BatchLvqSom":
        """Fit the batch LVQ-SOM refiner (not yet implemented).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : BatchLvqSom

        """
        raise NotImplementedError("fit() will be implemented in a later task")

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

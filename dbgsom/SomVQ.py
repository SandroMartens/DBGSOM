"""Implements the SOM Clusterer."""

import numpy as np
import numpy.typing as npt
from sklearn.base import (
    ClusterMixin,
    TransformerMixin,
    check_is_fitted,
)
from sklearn.utils.validation import validate_data

from dbgsom.BaseSom import BaseSom


class SomVQ(TransformerMixin, ClusterMixin, BaseSom):
    """Directed Batch Growing SOM for unsupervised clustering and vector quantization.

    See :class:`BaseSom` for all parameters.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster index of each training sample.

    som_ : networkx.Graph
        Graph containing neurons with ``weight``, ``error``, ``hit_count`` attributes.

    weights_ : ndarray of shape (n_prototypes, n_features)
        Learned prototype weight vectors.

    topographic_error_ : float
        Fraction of samples whose two nearest prototypes are not grid-adjacent.

    quantization_error_ : float
        Mean distance from each training sample to its nearest prototype.

    """

    def _label_prototypes(self, X: npt.ArrayLike, y=None) -> None:
        for i, neuron in enumerate(self.som_):
            self.som_.nodes[neuron]["label"] = i

    def predict(self, X: npt.ArrayLike) -> npt.NDArray:
        """Predict the closest neuron each sample in X belongs to.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Contiguous cluster index of the best matching prototype.

        """
        check_is_fitted(self, attributes=["weights_"])
        X = np.array(validate_data(self, X=X, reset=False))
        _, raw_winners = self._get_winning_neurons(X, n_bmu=1)
        return self._label_encoder[raw_winners]

    def _fit(self, X: npt.NDArray) -> None:
        _, raw_labels = self._get_winning_neurons(X, n_bmu=1)
        unique = np.unique(raw_labels)
        self._label_encoder = np.full(len(self.neurons_), -1, dtype=int)
        self._label_encoder[unique] = np.arange(len(unique))
        # Dead neurons: assign label of nearest live neuron by weight distance
        for d in np.where(self._label_encoder == -1)[0]:
            nearest = unique[
                np.argmin(
                    np.linalg.norm(self.weights_[unique] - self.weights_[d], axis=1)
                )
            ]
            self._label_encoder[d] = self._label_encoder[nearest]
        self.labels_ = self._label_encoder[raw_labels]

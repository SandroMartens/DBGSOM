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
    """A Directed Batch Growing Self-Organizing Map.

    This class implements the vector quantization/clustering functionality of the SOM.

    Parameters
    ----------
    n_iter : int, default = 200
        Maximum Number of training epochs.

    max_neurons : int, default = 100
        Maximum number of neurons in the som.

    vertical_growth : bool, default = False
        Wether to trigger hierarchical growth.

    decay_function : {'exponential', 'linear'}, default = 'exponential'
        Decay function to use for neighborhood bandwith sigma.

    verbose : bool, default = False

    coarse_training_frac : float, default = 0.5
        Fraction of max_iter to use for coarse training.

        Training happens in two phases, coarse and fine training. In coarse training,
        the neighborhood bandwidth is decreased from sigma_start to sigma_end and
        the network grows according to the growing rules. In fine training, the
        bandwidth is constant at sigma_end and no new neurons are added.

    growth_criterion : {"quantization_error", "entropy"}, default = "quantization_error"
        Method for calculating the error of neurons and samples.

        "quantization_error" : Use the quantization error of the prototypes.
        The cumulative error is the sum of individual errors of all samples.

        "entropy": For supervised learning we can use the entropy
        of labels of the samples represented by each prototype as error.

    metric : str, default = "euclidean"
        The metric to use for computing distances between prototypes and samples. Must
        be supported by scikit-learn or scipy.

    random_state : any (optional), default = None
        Random state for weight initialization.

    convergence_threshold : float, default = 10 ** -5
        If the sum of all weight changes is smaller than the threshold,
        convergence is assumed and the training is stopped.

    min_samples_vertical_growth : int, default = 100
        Minimum samples represented by a prototpye to trigger a vertical growth

    tau_2 : float, default = 0.5
        Global stopping criterion threshold for vertical growth (τ₂ in the GHSOM paper).
        A unit is expanded when its quantization error exceeds ``tau_2 * qe_0``.

    sigma_start, sigma_end : {None, numeric}, default = None
        Start and end value for the neighborhood bandwidth.

        If `None`, it is calculated dynamically in each epoch as

        `sigma_start = 0.2 * sqrt(n_neurons)`

        `sigma_end = max(0.7, 0.05 * sqrt(n_neurons))`

    **kwargs
        Additional parameters inherited from :class:`BaseSom`. See its
        documentation for ``neighborhood_function``,
        ``winner_stability_threshold``, ``pointer_search``,
        ``sigma_fine``, and others.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Labels of each point.

    som_ : NetworkX.graph
        Graph object containing the neurons with attributes.

    weights_ : ndarray of shape (n_prototypes, n_features)
        Learned weights of the neurons.

    topographic_error_ : float
        Fraction of training samples where the first and second best matching
        prototype are not neighbors on the SOM.

    quantization_error_ : float
        Average distance from all training samples to their nearest prototype.

    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags

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

"""Implements the SOM Classifier."""

from statistics import mode

import numpy as np
import numpy.typing as npt
from sklearn.base import (
    ClassifierMixin,
    TransformerMixin,
    check_is_fitted,
)
from sklearn.utils.validation import validate_data

from .BaseSom import BaseSom


class SomClassifier(TransformerMixin, ClassifierMixin, BaseSom):
    """A Directed Batch Growing Self-Organizing Map.

    This class implements the classification functionality of the SOM.

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
        be supported by sci-kit learn or scipy.

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
        Average distance from all training samples to their nearest prototypes.

    """

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> "SomClassifier":
        """Train SomClassifier on labelled data.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Training data.

        y : array_like of shape (n_samples,)
            Class labels. Required for the classifier.

        Returns
        -------
        self : SomClassifier
            Trained estimator.

        """
        if y is None:
            raise ValueError(
                f"{self.__class__.__name__} requires y to be passed, "
                "but the target y is None."
            )
        return super().fit(X, y)

    def _label_prototypes(self, X: npt.NDArray, y: npt.NDArray) -> None:
        """This method assigns labels to the prototypes based on the input data."""
        _, winners = self._get_winning_neurons(X, n_bmu=1)
        for winner_index, neuron in enumerate(self.neurons_):
            labels = y[winners == winner_index]
            # dead neuron
            if len(labels) == 0:
                self.som_.nodes[neuron]["label"] = -1
                self.som_.nodes[neuron]["probabilities"] = np.full(
                    shape=self.classes_.shape, fill_value=1.0 / len(self.classes_)
                )
                continue
            else:
                label_winner = mode(labels)
                labels, counts = np.unique(labels, return_counts=True)
            self.som_.nodes[neuron]["label"] = label_winner

            self.som_.nodes[neuron]["probabilities"] = np.zeros(
                shape=self.classes_.shape
            )
            hit_count = self.som_.nodes[neuron]["hit_count"]
            for class_id, count in zip(labels, counts):
                self.som_.nodes[neuron]["probabilities"][class_id] = (
                    count / hit_count if hit_count > 0 else 1
                )

    def predict(self, X: npt.ArrayLike) -> npt.NDArray:
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted class labels for samples in X.

        """
        check_is_fitted(self, attributes=["weights_"])
        X = validate_data(self, X, reset=False)
        labels = np.argmax(self.predict_proba(X=X), axis=1)
        return self.classes_[labels]

    def predict_proba(self, X: npt.ArrayLike, y: None = None) -> npt.NDArray:
        """Predict the probability of each class and each sample.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        y : Ignored. Only accepted for API compliance.

        Returns
        -------
        Probabilities: array of shape (n_samples, n_classes)

        Returns the probability of the sample for each class in the model, where
        classes are ordered as they are in self.classes_.

        """
        check_is_fitted(self, attributes=["weights_"])
        X = np.array(validate_data(self, X, reset=False))
        _, winners = self._get_winning_neurons(X, n_bmu=1)
        node_probabilities = self._extract_values_from_graph("probabilities")

        if not self.vertical_growth:
            return node_probabilities[winners]

        probabilities_rows = []
        for sample, winner in zip(X, winners):
            node = self.neurons_[winner]
            if "som" in self.som_.nodes[node]:
                child_som = self.som_.nodes[node]["som"]
                child_proba = child_som.predict_proba(sample.reshape(1, -1))[0]
                proba = np.zeros(len(self.classes_))
                child_indices = np.searchsorted(self.classes_, child_som.classes_)
                proba[child_indices] = child_proba
            else:
                proba = node_probabilities[winner]
            probabilities_rows.append(proba)
        return np.array(probabilities_rows)

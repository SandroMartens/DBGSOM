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
    """Directed Batch Growing SOM for supervised classification.

    See :class:`BaseSom` for all parameters.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Predicted class label of each training sample.

    classes_ : ndarray of shape (n_classes,)
        Unique class labels seen during fit.

    som_ : networkx.Graph
        Graph containing neurons with ``weight``, ``label``, ``probabilities`` attributes.

    weights_ : ndarray of shape (n_prototypes, n_features)
        Learned prototype weight vectors.

    topographic_error_ : float
        Fraction of samples whose two nearest prototypes are not grid-adjacent.

    quantization_error_ : float
        Mean distance from each training sample to its nearest prototype.

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

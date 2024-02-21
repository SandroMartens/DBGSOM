"""
Implements the SOM Classifier."""

from statistics import mode

import numpy as np
import numpy.typing as npt
from sklearn.base import (
    ClassifierMixin,
    TransformerMixin,
    check_array,
    check_is_fitted,
    check_X_y,
)

from dbgsom.BaseSom import BaseSom


class SomClassifier(BaseSom, TransformerMixin, ClassifierMixin):

    def _prepare_inputs(
        self, X: npt.ArrayLike, y=npt.ArrayLike
    ) -> tuple[npt.NDArray, npt.ArrayLike]:
        X, y = check_X_y(X=X, y=y, ensure_min_samples=4, dtype=[np.float64, np.float32])
        classes, y = np.unique(y, return_inverse=True)
        self.classes_ = np.array(classes)
        return X, y

    def _label_prototypes(self, X, y) -> None:
        winners = self._get_winning_neurons(X, n_bmu=1)
        for winner_index, neuron in enumerate(self.neurons_):
            labels = y[winners == winner_index]
            # dead neuron
            if len(labels) == 0:
                label_winner = -1
                labels = [-1]
                counts = [0]
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

    def _predict(self, X: npt.ArrayLike) -> np.ndarray:
        labels = np.argmax(self.predict_proba(X=X), axis=1)
        return labels

    def predict_proba(self, X: npt.ArrayLike) -> np.ndarray:
        """Predict the probability of each class and each sample.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        Probabilities: array of shape (n_samples, n_classes)

        Returns the probability of the sample for each class in the model, where
        classes are ordered as they are in self.classes_.
        """
        check_is_fitted(self)
        X = check_array(X)
        if self.vertical_growth:
            winners = self._get_winning_neurons(X, n_bmu=1)
            probabilities_rows = []
            for sample, winner in zip(X, winners):
                node = self.neurons_[winner]
                if "som" not in self.som_.nodes:
                    probabilities_sample = self.som_.nodes[node]["probabilities"]
                else:
                    probabilities_sample = self.som_.nodes[node]["som"].predict_proba(
                        sample
                    )

                probabilities_rows.append(probabilities_sample)

            probabilities = np.array(probabilities_rows)

        else:
            X_transformed = self.transform(X)
            probabilities = (
                X_transformed @ self._extract_values_from_graph("probabilities") / 50
            )

        return probabilities

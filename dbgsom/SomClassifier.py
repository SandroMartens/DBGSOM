"""
Implements the SOM Classifier."""

from statistics import mode
import numpy as np
import numpy.typing as npt
from dbgsom.BaseSom import BaseSom
from sklearn.base import (
    ClassifierMixin,
    TransformerMixin,
    check_X_y,
)


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

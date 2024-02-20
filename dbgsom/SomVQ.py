"""
Implements the SOM Clusterer."""

import numpy as np
import numpy.typing as npt
from dbgsom.BaseSom import BaseSom
from sklearn.base import (
    ClusterMixin,
    TransformerMixin,
    check_array,
)


class SomVQ(BaseSom, ClusterMixin, TransformerMixin):

    def _prepare_inputs(self, X: npt.ArrayLike, y=None) -> tuple[npt.NDArray, None]:
        X = check_array(array=X, ensure_min_samples=4, dtype=[np.float64, np.float32])
        return X, y

    def _predict(self, X: npt.ArrayLike) -> npt.NDArray:
        labels = self._get_winning_neurons(X, n_bmu=1)
        return labels

    def _label_prototypes(self, X: npt.ArrayLike, y=None) -> None:
        for i, neuron in enumerate(self.som_):
            self.som_.nodes[neuron]["label"] = i

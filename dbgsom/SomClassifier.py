"""
Implements the SOM Classifier."""

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

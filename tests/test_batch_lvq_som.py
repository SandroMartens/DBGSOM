import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils._param_validation import InvalidParameterError

from dbgsom.BatchLvqSom import BatchLvqSom


def test_predict_before_fit_raises():
    clf = BatchLvqSom()
    with pytest.raises(NotFittedError):
        clf.predict(np.zeros((3, 2)))


def test_param_constraints_reject_bad_n_iter():
    clf = BatchLvqSom(n_iter=0)
    with pytest.raises(InvalidParameterError):
        clf._validate_params()

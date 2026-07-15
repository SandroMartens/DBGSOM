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


class _FakeSom:
    """Minimal stand-in for a fitted SomVQ, isolating BatchLvqSom's own
    logic from SomVQ's growth/training cost in unit tests."""

    def __init__(self, weights, distance_matrix, sigma):
        self.weights_ = weights
        self._distance_matrix = distance_matrix
        self._sigma = sigma

    def _calculate_current_sigma(self):
        return self._sigma


_GRID_DM = np.array(
    [
        [0.0, 1.0, 1.0, 2.0],
        [1.0, 0.0, 2.0, 1.0],
        [1.0, 2.0, 0.0, 1.0],
        [2.0, 1.0, 1.0, 0.0],
    ]
)


def _brute_force_single_iteration(X, y_idx, weights, dm, sigma, n_classes):
    """Literal (non-vectorized) LVQ-SOM update, straight from the book
    formula. Used to regression-test the vectorized derivation in
    BatchLvqSom.fit -- the algebraic shortcut (gather + one matmul) is
    non-obvious enough to need an independent reference."""
    n_neurons, n_features = weights.shape
    diffs = X[:, None, :] - weights[None, :, :]
    winners = np.argmin(np.sum(diffs**2, axis=2), axis=1)

    node_labels = np.zeros(n_neurons, dtype=np.int64)
    for i in range(n_neurons):
        mask = winners == i
        if mask.any():
            counts = np.bincount(y_idx[mask], minlength=n_classes)
            node_labels[i] = np.argmax(counts)

    neighborhood = np.exp(-(dm**2) / (2 * sigma**2))
    new_weights = weights.copy()
    for i in range(n_neurons):
        numerator = np.zeros(n_features)
        denominator = 0.0
        for c in range(n_neurons):
            h = neighborhood[i, c]
            for s in np.where(winners == c)[0]:
                sign = 1.0 if y_idx[s] == node_labels[i] else -1.0
                numerator += h * sign * X[s]
                denominator += h * sign
        if denominator > 0:
            new_weights[i] = numerator / denominator
    return new_weights, node_labels


def test_fit_matches_brute_force_lvq_som_formula():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    y = rng.integers(0, 2, size=30)
    weights = rng.normal(size=(4, 3)) * 0.1
    sigma = 1.0
    som = _FakeSom(weights.copy(), _GRID_DM, sigma)

    clf = BatchLvqSom(n_iter=1, sigma=sigma).fit(X, y, som)

    classes, y_idx = np.unique(y, return_inverse=True)
    expected_weights, expected_labels = _brute_force_single_iteration(
        X, y_idx, weights.copy(), _GRID_DM, sigma, n_classes=len(classes)
    )
    np.testing.assert_allclose(clf.weights_, expected_weights, rtol=1e-10)
    np.testing.assert_array_equal(clf.node_labels_, classes[expected_labels])


def test_predict_returns_own_label_for_each_prototype():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    y = rng.integers(0, 2, size=30)
    weights = rng.normal(size=(4, 3)) * 0.1
    som = _FakeSom(weights, _GRID_DM, sigma=1.0)

    clf = BatchLvqSom(n_iter=3, sigma=1.0).fit(X, y, som)

    # Each prototype is its own nearest neighbor, so predicting on the
    # fitted weights_ must return exactly node_labels_.
    predictions = clf.predict(clf.weights_)
    np.testing.assert_array_equal(predictions, clf.node_labels_)


def test_fit_raises_if_som_not_fitted():
    class _UnfittedSom:
        pass

    clf = BatchLvqSom()
    with pytest.raises(NotFittedError):
        clf.fit(np.zeros((3, 2)), np.array([0, 1, 0]), _UnfittedSom())


def test_fit_raises_on_feature_mismatch():
    som = _FakeSom(
        weights=np.zeros((4, 5)),
        distance_matrix=_GRID_DM,
        sigma=1.0,
    )
    clf = BatchLvqSom()
    with pytest.raises(ValueError, match="features"):
        clf.fit(np.zeros((3, 2)), np.array([0, 1, 0]), som)

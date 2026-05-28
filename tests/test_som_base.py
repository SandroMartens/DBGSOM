import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.utils.estimator_checks import check_estimator

from dbgsom.SomClassifier import SomClassifier
from dbgsom.SomVQ import SomVQ


def test_scikit_learn_compatibility():
    """Prüft vollautomatisch alle Scikit-Learn API-Konventionen."""
    check_estimator(SomClassifier(), on_fail="warn")
    check_estimator(SomVQ(), on_fail="warn")


def test_som_mathematical_convergence():
    """Verifies that the SOM actually learns: QE must drop and the map must grow."""
    X, _ = load_digits(return_X_y=True)
    som = SomVQ(random_state=42, n_iter=50, max_neurons=30, verbose=False)

    initial_nodes = som.som_.number_of_nodes() if hasattr(som, "som_") else 0
    som.fit(X)

    assert som.som_.number_of_nodes() > initial_nodes, "SOM must grow during training"
    baseline_qe = np.linalg.norm(X - X.mean(axis=0), axis=1).mean()
    assert som.quantization_error_ < baseline_qe, (
        "QE must be lower than the naive single-centroid baseline"
    )
    assert 0.0 <= som.topographic_error_ <= 1.0


@pytest.mark.slow
def test_digits_training_regression():
    """Golden-value regression test: training on digits must produce stable results.

    Re-run this test after intentional algorithm changes to update the baselines.
    To regenerate: fit with the same parameters and print the three attributes below.
    """
    X, _ = load_digits(return_X_y=True)
    som = SomVQ(random_state=42, n_iter=50, max_neurons=30, verbose=False)
    som.fit(X)

    assert som.som_.number_of_nodes() == 24
    assert som.quantization_error_ == pytest.approx(24.09, abs=0.1)
    assert som.topographic_error_ == pytest.approx(0.117, abs=0.01)

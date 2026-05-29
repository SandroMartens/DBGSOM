import numpy as np
import pytest
from sklearn.datasets import load_digits, make_blobs
from sklearn.utils.estimator_checks import check_estimator

from dbgsom.SomClassifier import SomClassifier
from dbgsom.SomVQ import SomVQ


@pytest.fixture(scope="module")
def classifier_vq_pair():
    """SomVQ and SomClassifier trained on same blobs data with identical settings.

    cluster_std=0.3 ensures well-separated classes so each neuron is class-pure.
    """
    X, y = make_blobs(
        n_samples=300, centers=5, n_features=4, cluster_std=0.3, random_state=42
    )
    vq = SomVQ(random_state=42, n_iter=30, max_neurons=15, verbose=False).fit(X)
    clf = SomClassifier(random_state=42, n_iter=30, max_neurons=15, verbose=False).fit(
        X, y
    )
    return vq, clf, X, y


def test_cosine_metric_weights_are_unit_normalized():
    X, _ = make_blobs(n_samples=200, centers=5, n_features=4, random_state=42)
    vq = SomVQ(random_state=42, n_iter=30, max_neurons=15, metric="cosine").fit(X)
    norms = np.linalg.norm(vq.weights_, axis=1)
    np.testing.assert_array_almost_equal(norms, np.ones(len(norms)), decimal=6)


def test_cosine_metric_smoke():
    X, y = make_blobs(n_samples=200, centers=5, n_features=4, random_state=42)
    vq = SomVQ(random_state=42, n_iter=30, max_neurons=15, metric="cosine").fit(X)
    assert vq.som_.number_of_nodes() >= 1
    assert 0.0 <= vq.quantization_error_ <= 1.0
    assert 0.0 <= vq.topographic_error_ <= 1.0
    clf = SomClassifier(
        random_state=42, n_iter=30, max_neurons=15, metric="cosine"
    ).fit(X, y)
    assert 0.0 <= clf.score(X, y) <= 1.0


def test_classifier_entropy_criterion():
    X, y = make_blobs(
        n_samples=300, centers=5, n_features=4, cluster_std=0.3, random_state=42
    )
    clf = SomClassifier(
        random_state=42, n_iter=30, max_neurons=15, growth_criterion="entropy"
    )
    clf.fit(X, y)
    assert clf.som_.number_of_nodes() >= 4
    assert 0.0 <= clf.score(X, y) <= 1.0


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


def test_classifier_vq_identical_weights(classifier_vq_pair):
    """Same training → identical prototypes."""
    vq, clf, _, _ = classifier_vq_pair
    np.testing.assert_array_equal(vq.weights_, clf.weights_)


def test_classifier_vq_identical_bmu(classifier_vq_pair):
    """Identical weights → same best-matching unit for every sample."""
    vq, clf, X, _ = classifier_vq_pair
    _, vq_winners = vq._get_winning_neurons(X, n_bmu=1)
    _, clf_winners = clf._get_winning_neurons(X, n_bmu=1)
    np.testing.assert_array_equal(vq_winners, clf_winners)


def test_classifier_vq_identical_quantization_error(classifier_vq_pair):
    """Identical weights → same quantization error."""
    vq, clf, _, _ = classifier_vq_pair
    assert vq.quantization_error_ == pytest.approx(clf.quantization_error_)


def test_classifier_vq_identical_topographic_error(classifier_vq_pair):
    """Identical weights and topology → same topographic error."""
    vq, clf, _, _ = classifier_vq_pair
    assert vq.topographic_error_ == pytest.approx(clf.topographic_error_)


def test_classifier_prediction_matches_vq_majority(classifier_vq_pair):
    """Classifier labels each neuron with the majority class of its training samples.

    Tie-breaking: numpy argmax (lowest class index wins), matching the classifier's
    own argmax over stored probabilities.
    """
    vq, clf, X, y = classifier_vq_pair
    _, winners = vq._get_winning_neurons(X, n_bmu=1)
    node_probabilities = clf._extract_values_from_graph("probabilities")

    for neuron_idx in np.unique(winners):
        mask = winners == neuron_idx
        unique_classes, counts = np.unique(y[mask], return_counts=True)
        majority_class = unique_classes[np.argmax(counts)]
        predicted_class = clf.classes_[np.argmax(node_probabilities[neuron_idx])]
        assert predicted_class == majority_class, (
            f"Neuron {neuron_idx}: stored class doesn't match majority of its samples"
        )


@pytest.mark.slow
def test_digits_training_regression():
    """Golden-value regression test: training on digits must produce stable results.

    Re-run this test after intentional algorithm changes to update the baselines.
    To regenerate: fit with the same parameters and print the three attributes below.
    """
    X, y = load_digits(return_X_y=True)
    quantizer = SomVQ(random_state=42, n_iter=50, max_neurons=30, verbose=False)
    quantizer.fit(X)

    assert quantizer.som_.number_of_nodes() == 24
    assert quantizer.quantization_error_ == pytest.approx(24.09, abs=0.1)
    assert quantizer.topographic_error_ == pytest.approx(0.117, abs=0.01)

    clf = SomClassifier(random_state=42, n_iter=50, max_neurons=30, verbose=False)
    clf.fit(X, y)

    assert clf.som_.number_of_nodes() == 24
    assert clf.quantization_error_ == pytest.approx(24.09, abs=0.1)
    assert clf.topographic_error_ == pytest.approx(0.117, abs=0.01)
    assert clf.score(X, y) == pytest.approx(0.885, abs=0.01)
    np.testing.assert_almost_equal(
        desired=np.array(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    90.0,
                    90.0,
                    48.0,
                    34.0,
                    16.0,
                    4.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    -1.0,
                    -0.875,
                    -0.75,
                    -0.625,
                    -0.5,
                    -0.375,
                    -0.25,
                    -0.125,
                    0.0,
                    0.125,
                    0.25,
                    0.375,
                    0.5,
                    0.625,
                    0.75,
                    0.875,
                    1.0,
                ],
            ],
        ),
        actual=clf.topographic_function(X),
    )
    np.testing.assert_almost_equal(
        desired=np.array(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    90.0,
                    90.0,
                    48.0,
                    34.0,
                    16.0,
                    4.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    -1.0,
                    -0.875,
                    -0.75,
                    -0.625,
                    -0.5,
                    -0.375,
                    -0.25,
                    -0.125,
                    0.0,
                    0.125,
                    0.25,
                    0.375,
                    0.5,
                    0.625,
                    0.75,
                    0.875,
                    1.0,
                ],
            ],
        ),
        actual=quantizer.topographic_function(X),
    )

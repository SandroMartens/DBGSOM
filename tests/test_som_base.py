import pickle

import joblib
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
    check_estimator(SomClassifier(), on_fail="raise")
    check_estimator(SomVQ(), on_fail="raise")


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
    quantizer = SomVQ(random_state=42, n_iter=500, max_neurons=30, verbose=False)
    quantizer.fit(X)

    assert quantizer.som_.number_of_nodes() == 18
    assert quantizer.quantization_error_ == pytest.approx(22.93, abs=0.1)
    assert quantizer.topographic_error_ == pytest.approx(0.377, abs=0.01)

    clf = SomClassifier(random_state=42, n_iter=500, max_neurons=30, verbose=False)
    clf.fit(X, y)

    # assert clf.som_.number_of_nodes() == 26
    # assert clf.quantization_error_ == pytest.approx(24.09, abs=0.1)
    # assert clf.topographic_error_ == pytest.approx(0.117, abs=0.01)
    assert clf.score(X, y) == pytest.approx(0.895, abs=0.01)
    np.testing.assert_almost_equal(
        desired=np.array(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.05555556,
                    0.47222222,
                    0.41666667,
                    0.25925926,
                    0.16666667,
                    0.12037037,
                    0.02777778,
                    0.0,
                ],
                [
                    -1.0,
                    -0.83333333,
                    -0.66666667,
                    -0.5,
                    -0.33333333,
                    -0.16666667,
                    0.0,
                    0.16666667,
                    0.33333333,
                    0.5,
                    0.66666667,
                    0.83333333,
                    1.0,
                ],
            ]
        ),
        actual=quantizer.topographic_function(X),
        decimal=5,
    )


@pytest.mark.slow
def test_performance_path_matches_accuracy_path_quality():
    """Default fast path (pointer_search/cutgauss_phase='fine') must not sacrifice
    quantization quality vs. the textbook accuracy path ('none' for both)."""
    X, _ = load_digits(return_X_y=True)
    common = dict(random_state=42, n_iter=200, max_neurons=30, verbose=False)

    fast = SomVQ(**common).fit(X)
    accurate = SomVQ(pointer_search="none", cutgauss_phase="none", **common).fit(X)

    assert fast.quantization_error_ == pytest.approx(
        accurate.quantization_error_, rel=0.1
    )


def test_transform_output_shape(classifier_vq_pair):
    vq, _, X, _ = classifier_vq_pair
    result = vq.transform(X)
    assert result.shape == (len(X), vq.som_.number_of_nodes())


def test_transform_non_negative(classifier_vq_pair):
    vq, _, X, _ = classifier_vq_pair
    result = vq.transform(X)
    assert np.all(result >= 0)


def test_predict_proba_shape(classifier_vq_pair):
    _, clf, X, _ = classifier_vq_pair
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), len(clf.classes_))


def test_predict_proba_sums_to_one(classifier_vq_pair):
    _, clf, X, _ = classifier_vq_pair
    proba = clf.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-6)


def test_predict_proba_consistent_with_predict(classifier_vq_pair):
    _, clf, X, _ = classifier_vq_pair
    proba = clf.predict_proba(X)
    argmax_labels = clf.classes_[np.argmax(proba, axis=1)]
    np.testing.assert_array_equal(argmax_labels, clf.predict(X))


def test_pickle_roundtrip(classifier_vq_pair):
    vq, clf, X, _ = classifier_vq_pair
    for estimator in (vq, clf):
        blob = pickle.dumps(estimator)
        restored = pickle.loads(blob)
        np.testing.assert_array_equal(estimator.predict(X), restored.predict(X))


def test_joblib_roundtrip(classifier_vq_pair, tmp_path):
    vq, _, X, _ = classifier_vq_pair
    path = tmp_path / "vq.joblib"
    joblib.dump(vq, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(vq.predict(X), restored.predict(X))


def test_determinism_same_seed():
    X, _ = make_blobs(n_samples=200, centers=4, n_features=3, random_state=0)
    vq1 = SomVQ(random_state=7, n_iter=20, max_neurons=10).fit(X)
    vq2 = SomVQ(random_state=7, n_iter=20, max_neurons=10).fit(X)
    np.testing.assert_array_equal(vq1.weights_, vq2.weights_)


def test_different_seeds_differ():
    X, _ = make_blobs(n_samples=200, centers=4, n_features=3, random_state=0)
    vq1 = SomVQ(random_state=1, n_iter=20, max_neurons=10).fit(X)
    vq2 = SomVQ(random_state=99, n_iter=20, max_neurons=10).fit(X)
    assert not np.array_equal(vq1.weights_, vq2.weights_)


def test_verbose_no_exception():
    X, _ = make_blobs(n_samples=100, centers=4, n_features=2, random_state=0)
    SomVQ(random_state=0, n_iter=10, max_neurons=8, verbose=True).fit(X)


def test_max_neurons_limit_respected():
    X, _ = make_blobs(n_samples=300, centers=10, n_features=4, random_state=0)
    limit = 8
    vq = SomVQ(random_state=0, n_iter=50, max_neurons=limit).fit(X)
    assert vq.som_.number_of_nodes() <= limit


def test_state_sync_invariant_after_growth():
    """neurons_, _node_to_idx and weights_ must stay in lockstep with som_ after growth."""
    X, _ = load_digits(return_X_y=True)
    som = SomVQ(random_state=42, n_iter=50, max_neurons=30, verbose=False).fit(X)

    assert som.som_.number_of_nodes() > 1, "test requires growth to have happened"
    assert len(som.neurons_) == len(som._node_to_idx) == som.weights_.shape[0]
    assert len(som.neurons_) == som.som_.number_of_nodes()
    assert set(som.neurons_) == set(som.som_.nodes())
    for idx, node in enumerate(som.neurons_):
        assert som._node_to_idx[node] == idx

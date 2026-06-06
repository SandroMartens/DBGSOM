import numpy as np
import pytest

from dbgsom._kernels import (
    numba_quantization_error,
    numba_voronoi_set_centers,
)


# Eine einfache, sichere Referenzfunktion in purem NumPy
def numpy_reference_error(winners, length, distances):
    # np.bincount macht genau das, was deine Schleife tut, nur in nativem C/NumPy
    return np.bincount(winners, weights=distances, minlength=length)


def test_simple_unique_winners():
    """Testet den Fall, dass jeder Winner nur einmal vorkommt."""
    winners = np.array([0, 1, 2], dtype=np.int64)
    distances = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    length = 4

    expected = np.array([1.5, 2.5, 3.5, 0.0])
    result = numba_quantization_error(winners, length, distances)

    np.testing.assert_array_almost_equal(result, expected)


def test_duplicate_winners():
    """Testet, ob Distanzen für denselben Winner korrekt aufaddiert werden."""
    winners = np.array([1, 1, 2, 1], dtype=np.int64)
    distances = np.array([1.0, 2.0, 1.5, 0.5], dtype=np.float64)
    length = 4

    # Index 1 sollte: 1.0 + 2.0 + 0.5 = 3.5 sein
    expected = np.array([0.0, 3.5, 1.5, 0.0])
    result = numba_quantization_error(winners, length, distances)

    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize("seed", [42, 1337, 2026])
def test_against_numpy_reference(seed):
    """Vergleicht die Numba-Funktion mit einer validen NumPy-Referenz bei großen Zufallsdaten."""
    rng = np.random.default_rng(seed)

    size = 1000
    length = 50

    winners = rng.integers(0, length, size=size)
    distances = rng.uniform(0.0, 10.0, size=size)

    expected = numpy_reference_error(winners, length, distances)
    result = numba_quantization_error(winners, length, distances)

    np.testing.assert_array_almost_equal(result, expected)


def naive_voronoi_set_centers(data, winners, sample_weights, shape):
    """Einfache Referenz-Implementierung mittels intuitivem Boolean Masking.

    Tut exakt dasselbe wie die optimierte Numba-Version, nur ohne Performance-Voodoo.
    """
    # Initialisiere das Array für die neuen Zentren (Form der Gewichtsmatrix, z.B. [n_neurons, n_features])
    voronoi_centers = np.zeros(shape)

    # Wir gehen stur jede mögliche Neuron-ID durch
    for neuron_id in range(shape[0]):
        # Maske: Welche Datenpunkte haben dieses Neuron als Winner?
        mask = winners == neuron_id

        # Wenn diesem Neuron überhaupt Datenpunkte zugewiesen wurden
        if np.any(mask):
            neuron_samples = data[mask]
            neuron_weights = sample_weights[mask]

            # Gewichteten Mittelwert berechnen
            # np.newaxis wird benötigt, damit die 1D-Gewichte korrekt mit den 2D-Daten multipliziert werden
            weighted_sum = np.sum(
                neuron_samples * neuron_weights[:, np.newaxis], axis=0
            )
            total_weight = np.sum(neuron_weights)

            voronoi_centers[neuron_id] = weighted_sum / total_weight
        else:
            # Wenn das Neuron leer ausging (kein Datenpunkt im Voronoi-Set),
            # bleibt es im Batch-Schritt üblicherweise auf 0 (oder behält den alten Wert)
            voronoi_centers[neuron_id] = 0.0

    return voronoi_centers


def test_numba_voronoi_against_naive_oracle():
    # 1. Kleine, überschaubare Testdaten generieren (5 Samples, 3 Features)
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [1.5, 2.5, 3.5],
            [8.0, 9.0, 10.0],
        ]
    )

    # Zufällige Wichtigkeits-Gewichte für die Samples
    sample_weights = np.array([1.0, 2.0, 1.5, 0.5, 1.0])

    # Zuweisung zu insgesamt 4 Neuronen (IDs: 0, 1, 2, 3)
    winners = np.array([0, 2, 0, 1, 2])

    # Form der SOM-Gewichtsmatrix (4 Neuronen, 3 Features)
    weights_shape = (4, 3)

    # 2. DIE REFERENZ (Der alte/neue einfache Code)
    expected_centers = naive_voronoi_set_centers(
        data=data, winners=winners, sample_weights=sample_weights, shape=weights_shape
    )

    # 3. DEIN OPTIMIERTER CODE
    # (Hier simulieren wir den Ablauf aus deiner Pipeline)
    index = np.argsort(winners)
    groups, offsets = np.unique(winners[index], return_index=True)

    actual_centers = numba_voronoi_set_centers(
        kernel=sample_weights,
        data=data,
        shape=weights_shape,
        groups=groups,
        offsets=offsets,
        index=index,
    )

    # 4. DER MATHEMATISCHE ABGLEICH
    # Erlaubt minimale Rundungsfehler durch Floats (decimal=5)
    np.testing.assert_array_almost_equal(actual_centers, expected_centers, decimal=5)

import numpy as np
import pytest
from dbgsom.BaseSom import (
    numba_quantization_error,
)  # Ersetze 'your_module' mit deinem Dateinamen


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

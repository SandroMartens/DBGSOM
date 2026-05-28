from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from dbgsom.SomVQ import SomVQ


def test_topographic_function_matrix_logic():
    """Testet die logischen Verknüpfungen von Euclid, Chebyshev und Delaunay."""
    # Wir erstellen 3 Neuronen in einer Reihe, um Abstände zu kontrollieren
    # N0 bei (0,0), N1 bei (1,0), N2 bei (2,0)
    neurons = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    som = SomVQ()
    som.neurons_ = neurons

    # Floyd-Warshall graph distances for a 3-node line N0-N1-N2:
    # [[0, 1, 2],
    #  [1, 0, 1],
    #  [2, 1, 0]]
    som._distance_matrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)

    # Wir faken die Delaunay-Matrix (Form 3x3)
    # Symmetrisch: Verbindung zwischen 0-1 und 1-2
    mock_delaunay = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    som._calculate_delaunay_triangulation = MagicMock(return_value=mock_delaunay)

    # Ausführung (Input X ist hier egal, da Delaunay gemockt ist)
    result = som.topographic_function(X=np.array([[0, 0]]))

    # max_dist zwischen N0 und N2 ist 2.
    # k_values läuft also von -2 bis +2 -> Länge 5
    assert result.shape == (2, 5)

    k_values = np.array([-2, -1, 0, 1, 2])
    phi_values = result[0]

    # --- MANUELLE VERIFIKATION DER LOGIK ---
    # k = 2 (>0): max_dist > 2 UND delaunay == 1.
    #             Gibt es nicht (Max dist ist 2, nicht größer als 2). -> Erwartet: 0
    assert phi_values[4] == 0

    # k = 1 (>0): max_dist > 1 UND delaunay == 1.
    #             Gibt es nicht, da dort wo delaunay == 1 ist, die Distanz exakt 1 ist (nicht > 1). -> Erwartet: 0
    assert phi_values[3] == 0

    # k = -1 (<0): euclid == 1 UND delaunay > 1.
    #              Dort wo euclid == 1 ist (Nachbarn), ist delaunay exakt 1 (nicht > 1). -> Erwartet: 0
    assert phi_values[1] == 0

    # k = 0: Muss die Summe aus _phi(-1) und _phi(1) sein -> 0 + 0 = 0
    assert phi_values[2] == 0

    # Test bestanden, wenn die berechnete Kurve exakt der mathematischen Erwartung entspricht
    np.testing.assert_array_equal(phi_values, np.array([0, 0, 0, 0, 0]))


class TestQuantizationError:
    def test_quantization_error_correctness(self):
        """Prüft die mathematische Korrektheit gegen ein handberechnetes Ergebnis."""
        # Setup einer minimalen SOM
        som = SomVQ()

        # Wir simulieren den Zustand nach dem Fit (State Injection)
        som.weights_ = np.array(
            [
                [0.0, 0.0],  # Neuron 0
                [10.0, 10.0],  # Neuron 1
            ]
        )

        # Dummy-Implementierung für _get_winning_neurons motzen,
        # damit wir exakte Test-Distanzen zurückbekommen
        # Punkt 1 hat Abstand 2 zu Neuron 0, Punkt 2 hat Abstand 1 zu Neuron 1
        def mock_winning_neurons(X, n_bmu=1):
            # Wir geben fest definierte Distanzen zurück:
            # Sample 1: Distanz 2.0 | Sample 2: Distanz 1.0
            return np.array([2.0, 1.0]), np.array([0, 1])

        som._get_winning_neurons = mock_winning_neurons

        # Test-Daten (Inhalt ist wegen Mock egal, Shape muss stimmen)
        X_test = np.array([[1.0, 1.0], [9.0, 9.0]])

        # Erwarteter Fehler: (2.0 + 1.0) / 2 = 1.5
        expected_error = 1.5
        actual_error = som.calculate_quantization_error(X_test)

        assert actual_error == pytest.approx(expected_error)
        assert isinstance(actual_error, float)

    def test_raises_not_fitted_error(self):
        """Stellt sicher, dass die Funktion ohne vorheriges Training blockiert."""
        som = SomVQ()
        X_test = np.array([[1.0, 2.0]])

        # Da fit() nicht aufgerufen wurde, MUSS scikit-learns NotFittedError fliegen
        with pytest.raises(NotFittedError):
            som.calculate_quantization_error(X_test)

    def test_invalid_input_nan_raises_error(self):
        """Prüft, ob fehlerhafte Daten (NaNs) dank check_array abgefangen werden."""
        som = SomVQ()
        som.weights_ = np.array([[0.0, 0.0]])  # Als "fitted" simulieren

        # Daten mit einem unzulässigen NaN-Wert
        invalid_data = np.array([[1.0, np.nan]])

        # check_array sollte hier einen ValueError werfen
        with pytest.raises(ValueError, match="Input X contains NaN"):
            som.calculate_quantization_error(invalid_data)


class TestTopographicError:
    @pytest.fixture
    def fitted_som(self):
        """Erstellt eine simulierte, fertig trainierte SOM mit 4 Neuronen auf einem Gitter."""
        som = SomVQ()

        # Wir simulieren ein 2x2 Gitter von Neuronen (Koordinaten im Raum)
        # Neuron 0: (0,0) | Neuron 1: (0,1)
        # Neuron 2: (1,0) | Neuron 3: (1,1)
        som.neurons_ = np.array(
            [
                [0.0, 0.0],  # ID 0
                [0.0, 1.0],  # ID 1
                [1.0, 0.0],  # ID 2
                [1.0, 1.0],  # ID 3
            ]
        )

        # Ein Alibi-weights_ Attribut setzen, damit check_is_fitted() durchgeht
        som.weights_ = np.random.rand(4, 3)
        return som

    def test_topographic_error_perfect_topology(self, fitted_som):
        """Testet den Fall, dass alle BMU-Nachbarn direkt verbunden sind (Error = 0.0)."""
        # Mock für _get_winning_neurons:
        # Sample 1 wählt Neuron 0 & 1 (Abstand 1.0 -> verbunden)
        # Sample 2 wählt Neuron 2 & 3 (Abstand 1.0 -> verbunden)
        fitted_som._get_winning_neurons = lambda X, n_bmu: (
            None,
            np.array([[0, 1], [2, 3]]),
        )

        X_test = np.array([[1, 2, 3], [4, 5, 6]])
        error = fitted_som._calculate_topographic_error(X_test)

        assert error == pytest.approx(0.0)

    def test_topographic_error_broken_topology(self, fitted_som):
        """Testet den Fall, dass die BMUs weit auseinanderliegen (Error = 1.0)."""
        # Sample 1 wählt Neuron 0 & 3 (Diagonaler Abstand ist sqrt(2) ≈ 1.414.
        # Wenn Schwellenwert 1.5 ist, ist das noch okay. Wir erzwingen ID 0 und eine fiktive ID weit weg)
        # Um den Test sicher zu machen, fälschen wir die Gitter-Distanzen im Mock-Objekt:
        fitted_som.neurons_ = np.array([[0.0, 0.0], [0.0, 10.0]])  # Abstand ist 10.0

        fitted_som._get_winning_neurons = lambda X, n_bmu: (None, np.array([[0, 1]]))

        X_test = np.array([[1, 2, 3]])
        error = fitted_som._calculate_topographic_error(X_test)

        assert error == pytest.approx(1.0)

    def test_topographic_error_mixed_topology(self, fitted_som):
        """Testet ein 50/50 Szenario (Ein Sample korrekt, eins fehlerhaft -> Error = 0.5)."""
        # Wir modifizieren das Gitter so, dass Abstand von 0 zu 1 = 1.0 ist, von 0 zu 2 = 5.0
        fitted_som.neurons_ = np.array(
            [
                [0.0, 0.0],  # 0
                [0.0, 1.0],  # 1 (nah an 0)
                [0.0, 5.0],  # 2 (weit weg von 0)
            ]
        )

        # Sample 1: [0, 1] -> Nah (kein Fehler)
        # Sample 2: [0, 2] -> Weit (Fehler)
        fitted_som._get_winning_neurons = lambda X, n_bmu: (
            None,
            np.array([[0, 1], [0, 2]]),
        )

        X_test = np.array([[1, 2, 3], [4, 5, 6]])
        error = fitted_som._calculate_topographic_error(X_test)

        assert error == pytest.approx(0.5)

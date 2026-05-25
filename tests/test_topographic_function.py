import numpy as np
from unittest.mock import MagicMock
from dbgsom.BaseSom import BaseSom  # Ersetze 'your_module' mit deinem Dateinamen


def test_topographic_function_matrix_logic():
    """Testet die logischen Verknüpfungen von Euclid, Chebyshev und Delaunay."""

    # Wir erstellen 3 Neuronen in einer Reihe, um Abstände zu kontrollieren
    # N0 bei (0,0), N1 bei (1,0), N2 bei (2,0)
    neurons = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    som = BaseSom()
    som.neurons_ = neurons

    # Wir faken die Delaunay-Matrix (Form 3x3)
    # Symmetrisch: Verbindung zwischen 0-1 und 1-2
    mock_delaunay = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    som._calculate_delaunay_triangulation = MagicMock(return_value=mock_delaunay)

    # Zur Erinnerung, was intern passiert:
    # Max-Distanzen (Chebyshev) zwischen den Neuronen:
    # [[0, 1, 2],
    #  [1, 0, 1],
    #  [2, 1, 0]]
    #
    # Euclid-Distanzen: Identisch zu Chebyshev in diesem 1D-Fall.

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

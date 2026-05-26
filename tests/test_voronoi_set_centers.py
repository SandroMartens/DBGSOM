import numpy as np

from dbgsom.BaseSom import numba_voronoi_set_centers


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

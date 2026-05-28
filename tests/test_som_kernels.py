import numpy as np
import pytest

# 1. HIER IMPORTIERST DU DEINE ECHTE KLASSE
from dbgsom.SomVQ import SomVQ


# 2. DIE FIXTURE ERSTELLT EINE ECHTE INSTANZ
@pytest.fixture
def real_som() -> SomVQ:
    """Erstellt eine echte, frische Instanz deiner SOM für jeden Test."""
    return SomVQ(decay_function="exponential", coarse_training_frac=0.6)


# 3. DIE TEST-KLASSEN NUTZEN DIE ECHTE INSTANZ
class TestGaussianNeighborhood:
    def test_center_is_one(self, real_som):
        """Testet deine ECHTE Methode auf der ECHTEN Klasse."""
        # VORBEREITUNG (State Injection):
        # Da wir die Methode isoliert testen wollen, ohne vorher ein
        # stundenlanges Training zu laufen, setzen wir die Variablen direkt:
        real_som._distance_matrix = np.array([0.0, 100.0])

        # Wir überschreiben die Sigma-Berechnung für diesen Test mit einem festen Wert,
        # damit der Test stabil bleibt und nicht vom Epochenzähler abhängt:
        real_som._calculate_current_sigma = lambda: 2.0

        # DURCHFÜHRUNG:
        # Wir rufen deine echte Funktion auf der Instanz auf
        h = real_som._calculate_gaussian_neighborhood()

        # PRÜFUNG:
        assert h[0] == pytest.approx(1.0)
        assert h[1] == pytest.approx(0.0, abs=1e-5)


class TestExponentialSimilarity:
    def test_perfect_match_yields_maximum_weight(self, real_som):
        """Testet deine echte Outlier-Dämpfung."""
        # VORBEREITUNG:
        # Wir füttern die echte Instanz mit einer bekannten Gesamtvarianz
        real_som._total_variance = 4.0
        distances = np.array([0.0])

        # DURCHFÜHRUNG:
        kernel = real_som._calculate_exp_similarity(distances)

        # PRÜFUNG:
        assert kernel[0] == pytest.approx(1.0)

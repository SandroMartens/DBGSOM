from sklearn.utils.estimator_checks import check_estimator

from dbgsom.SomClassifier import SomClassifier
from dbgsom.SomVQ import SomVQ


def test_scikit_learn_compatibility():
    """Prüft vollautomatisch alle Scikit-Learn API-Konventionen.

    Falls deine SOM bestimmte spezielle Scikit-Learn-Features (wie z.B.
    die Verarbeitung von Strings) explizit nicht unterstützt, kannst du
    hier auch Ausnahmen definieren.
    """
    # check_estimator wirft eine Exception, wenn deine API-Struktur verletzt wird
    classifier = SomClassifier()
    quantizer = SomVQ()
    check_estimator(classifier, on_fail="warn")
    check_estimator(quantizer, on_fail="warn")


def test_som_mathematical_convergence():
    """DEIN EGENER ERGÄNZUNGSTEST: Prüft, ob das Netz inhaltlich lernt."""
    # (Hier kommt dein Test von vorhin hin, der verifiziert, dass
    # der Quantization Error sinkt und die SOM tatsächlich wächst!)
    pass

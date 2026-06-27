# Design: Lean-Spec + Property-Based Test für State-Sync-Invariante

Datum: 2026-06-27
Status: Entwurf — wartet auf Review

## Hintergrund

Commit `7e8cca9` hat erstmals die State-Sync-Invariante getestet: `neurons_`,
`_node_to_idx` und `weights_` müssen nach jedem Wachstumsschritt mit dem
Graphen `som_` im Lockstep bleiben (Schreibstelle: `_add_node_to_graph`,
siehe `CLAUDE.md`). Der bestehende Test prüft das nur an EINEM
Beispiel-Wachstumslauf (∃-Aussage). Ziel dieses Projekts: dieselbe
Invariante als ∀-Aussage absichern — formal in Lean formuliert, als
Property-Based-Test gegen den echten `BaseSom`-Code ausgeführt.

Kontext: experimenteller Vorstoß, ob/wie Lean als Spezifikationssprache für
Python-Testhärtung in diesem Repo Sinn ergibt — siehe vorangegangene
Diskussion (Lean vs. Haskell, Drift-Problem bei KI-generierten Tests).
Gewählter Ansatz: Lean-Spec liefert eindeutige Formulierung + trivialen
Beweis auf einem abstrakten Modell; keine automatische Code-Generierung,
keine Laufzeitkopplung zu Python.

## Ziel

Property-Based-Test, der für beliebige (Hypothesis-generierte) Datensätze
und Seeds prüft:

Nach **jedem** Knoten-Insertion-Event während `fit()` (Schreibstelle
`_add_node_to_graph`, BaseSom.py:1248-1255 — `som_.add_node`,
`_node_to_idx[node]` und `neurons_.append(node)` laufen dort synchron):

1. `len(neurons_) == len(_node_to_idx)`
2. `∀ i: _node_to_idx[neurons_[i]] == i`
3. `set(neurons_) == set(som_.nodes())`

Nach Ende von `fit()` zusätzlich (entspricht bestehendem Test `7e8cca9`,
jetzt über viele Datensätze/Seeds statt einem Beispiel):

1. `weights_.shape[0] == len(neurons_)`

`weights_` wird laut `_grow_som`/CLAUDE.md erst **am nächsten Epochenstart**
aus dem Graphen neu extrahiert, nicht synchron in `_add_node_to_graph` —
Prüfung 4 darf deshalb nur nach abgeschlossenem `fit()` laufen, nicht nach
jedem Insertion-Event (sonst False Positives durch den absichtlichen Lag).

Lean-Spec dokumentiert dieselbe Eigenschaft unzweideutig + beweist sie für
ein abstraktes, atomares Insertions-Modell.

## Out of Scope

- Keine Lean→Python-Codegenerierung.
- Keine CI-Integration des Lean-Buildsteps (rein lokal/dokumentarisch).
- Keine Differential-Tests gegen extrahierten Lean-Code.
- Keine weiteren Invarianten (Growing-Symmetrie, BMU-Kernel-Parität,
  Weight-Update-Richtung) — eigene Folge-Iterationen falls gewünscht.
- Keine Abdeckung der Fine-Phase-Sonderfälle (kein Growth dort per Design).

## Architektur

```
formal/
  lakefile.toml
  lean-toolchain
  Spec/StateSync.lean      -- abstrakte SOMState + Invariante + Beweis
tests/
  strategies.py             -- Hypothesis-Strategien
  test_state_sync_property.py
pyproject.toml               -- + hypothesis als dev-dependency
```

`formal/` ist vollständig von `dbgsom/` getrennt: kein Import, kein Build-
Schritt im Package, keine Laufzeitkopplung. Lean-Datei ist Dokumentation +
Beweis, kein ausführbarer Teil der Test-Pipeline.

## Lean-Spec

Abstraktes Modell von `SOMState` mit den drei relevanten Feldern
(Knotenliste, Index-Map, Gewichtsmatrix-Zeilenzahl) und einer Insertion,
die alle drei atomar zusammen aktualisiert (Modell von
`_add_node_to_graph`):

```lean
structure SOMState (n d : ℕ) where
  neurons     : Fin n → Node
  nodeToIdx   : Node → Option (Fin n)
  weightsRows : ℕ

def StateSync (s : SOMState n d) : Prop :=
  s.weightsRows = n ∧
  ∀ i : Fin n, s.nodeToIdx (s.neurons i) = some i

def insertNode (s : SOMState n d) (newNode : Node) :
    SOMState (n + 1) d :=
  { neurons     := Fin.snoc s.neurons newNode
    nodeToIdx   := fun node => if node = newNode
                                then some (Fin.last n)
                                else (s.nodeToIdx node).map Fin.castSucc
    weightsRows := s.weightsRows + 1 }

theorem insert_preserves_sync (s : SOMState n d) (newNode : Node)
    (h : StateSync s) : StateSync (insertNode s newNode) := by
  constructor
  · simp [insertNode, h.1]
  · intro i
    sorry  -- Fallunterscheidung i = last vs. i = castSucc j, nutzt h.2
```

(Beweis-Skelett — Detailausarbeitung Teil der Implementierung, kein
offenes Forschungsproblem, da Insertion per Definition atomar ist.)

Wert der Lean-Spec liegt in der **eindeutigen Formulierung** der
Invariante, nicht in Beweistiefe — Übersetzung der Prop-Aussage in die
Hypothesis-Property im nächsten Abschnitt ist 1:1.

## Python Property-Test

Hook in `_add_node_to_graph` per Monkeypatch, um die Invariante nach
**jedem** Insertion-Event zu prüfen (nicht nur am Endzustand nach `fit()`):

```python
# tests/strategies.py
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np

@st.composite
def growable_dataset(draw):
    n_features = draw(st.integers(2, 5))
    n_samples = draw(st.integers(20, 60))
    X = draw(arrays(
        np.float64, (n_samples, n_features),
        elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
    ))
    seed = draw(st.integers(0, 10_000))
    return X, seed
```

```python
# tests/test_state_sync_property.py
from unittest.mock import patch
from hypothesis import given, settings

from dbgsom import SomVQ
from dbgsom.BaseSom import BaseSom
from strategies import growable_dataset


@given(growable_dataset())
@settings(max_examples=20, deadline=None)
def test_state_sync_holds_after_every_growth_event(data):
    X, seed = data
    violations = []
    original = BaseSom._add_node_to_graph

    def wrapped(self, *args, **kwargs):
        original(self, *args, **kwargs)
        idx_ok = all(
            self._node_to_idx[node] == i
            for i, node in enumerate(self.neurons_)
        )
        if (
            len(self.neurons_) != len(self._node_to_idx)
            or not idx_ok
            or set(self.neurons_) != set(self.som_.nodes())
        ):
            violations.append(self.som_.number_of_nodes())

    with patch.object(BaseSom, "_add_node_to_graph", wrapped):
        som = SomVQ(random_state=seed).fit(X)

    assert not violations, f"state desync at node counts {violations}"
    assert som.weights_.shape[0] == len(som.neurons_)
```

Hypothesis shrinkt fehlschlagende Fälle automatisch auf minimales
`(X, seed)`. `max_examples=20`, `deadline=None` (Numba-JIT-Warmup pro
Beispiel) — bleibt innerhalb `pytest -m "not slow"`-Budget. Import
`from strategies import growable_dataset` (kein `tests.`-Präfix): `tests/`
hat kein `__init__.py`, pytest fügt im Rootless-Modus das Testverzeichnis
selbst zu `sys.path` hinzu.

## Dependencies

`pyproject.toml`: `hypothesis` in `[dependency-groups] dev` (keine
Laufzeit-Dependency für `dbgsom` selbst).

`formal/`: separates Lean-4-Projekt, lokal per `lake build` prüfbar, kein
CI-Job in dieser Iteration.

## Bekannte Grenzen

- Lean-Beweis und Python-Test sind unabhängig — kein Mechanismus erzwingt
  Sync zwischen beiden bei künftigen Code-Änderungen. Falls
  `_add_node_to_graph` sich strukturell ändert, muss die Lean-Spec manuell
  nachgezogen werden.
- Hypothesis approximiert ∀ durch Sampling (20 Beispiele), ist kein
  vollständiger Beweis für den echten Python-Code — nur die abstrakte
  Lean-Spec ist vollständig bewiesen.
- Monkeypatch-Hook ist an die aktuelle Schreibstelle (`_add_node_to_graph`)
  gebunden; verschiebt sich die Schreibstelle, muss der Test angepasst
  werden.

## Testing

- Neuer Property-Test läuft per Default mit `pytest -m "not slow"` (kein
  `slow`-Marker nötig bei `max_examples=20` + kleinen Datensätzen).
- Lean-Seite: `lake build` lokal, kein automatisierter Test-Runner-Bezug.

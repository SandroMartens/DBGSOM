from unittest.mock import patch

from hypothesis import given, settings
from strategies import growable_dataset

from dbgsom.BaseSom import BaseSom
from dbgsom.SomVQ import SomVQ


@given(growable_dataset())
@settings(max_examples=20, deadline=None)
def test_state_sync_holds_after_every_growth_event(data):
    """neurons_/_node_to_idx/som_.nodes() must stay in lockstep after each
    insertion event; weights_ syncs only once fit() has finished growing."""
    X, seed = data
    violations = []
    original = BaseSom._add_node_to_graph

    def wrapped(self, *args, **kwargs):
        original(self, *args, **kwargs)
        idx_ok = all(
            self._node_to_idx[node] == i for i, node in enumerate(self.neurons_)
        )
        if (
            len(self.neurons_) != len(self._node_to_idx)
            or not idx_ok
            or set(self.neurons_) != set(self.som_.nodes())
        ):
            violations.append(self.som_.number_of_nodes())

    with patch.object(BaseSom, "_add_node_to_graph", wrapped):
        som = SomVQ(random_state=seed, n_iter=50, max_neurons=20, lambda_=2.0).fit(X)

    assert not violations, f"state desync at node counts {violations}"
    assert som.weights_.shape[0] == len(som.neurons_)

## Batch LVQ-SOM — Design

### Context

`dbgsom` currently offers unsupervised vector quantization (`SomVQ`) and a
supervised classifier that grows its own topology (`SomClassifier`). Kohonen
(2001) describes a third mode in section 6.11 ("The LVQ-SOM"): take a
converged SOM topology and refine it with a supervised, LVQ-style sign rule
applied through the existing neighborhood kernel. Section 6.4 gives the
batch form of plain LVQ1 (single-node update, no neighborhood fan-out); 6.11
generalizes it to the full neighborhood, which is the variant implemented
here.

Kohonen recommends running this only after an unsupervised SOM phase, once
the neighborhood radius has shrunk to its final value — i.e. sigma is fixed,
not decayed, during this refinement.

### Non-goals

- No topology growth. Operates on a fixed, pre-trained `SomVQ` graph.
- No sigma decay schedule — single fixed sigma for the whole refinement.
- No convergence check — fixed `n_iter`, per the book's "repeat a few
  times."
- Not a drop-in sklearn `Pipeline`/`GridSearchCV` estimator (`fit` takes an
  extra required `som` argument, breaking the `fit(X, y)` convention).

### API

```python
class BatchLvqSom(ClassifierMixin, BaseEstimator):
    def __init__(self, n_iter: int = 10, sigma: float | None = None): ...

    def fit(self, X, y, som: SomVQ) -> "BatchLvqSom":
        """Refine a fitted SomVQ's prototypes with the batch LVQ-SOM rule.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        som : SomVQ
            Already-fitted instance. Supplies initial weights, graph
            topology, and distance matrix. Not mutated.
        """

    def predict(self, X) -> np.ndarray: ...
```

Fitted attributes: `weights_`, `node_labels_`, `classes_` — naming matches
`SomVQ`/`SomClassifier` conventions.

`som` must already be fitted (`check_is_fitted(som, "weights_")`). Its
`weights_` array is copied, not mutated in place.

Default `sigma=None` resolves to `som._calculate_current_sigma()` — the
fine-phase sigma value in effect at the end of the SOM's own training.

### Algorithm

Notation: `K` = number of neurons, `C` = number of classes, `D` = number of
features, `dm` = `som._distance_matrix` (graph hop counts, cast to
`float64`).

Neighborhood kernel is computed **once**, before the iteration loop —
topology and sigma are both fixed for the whole refinement, unlike
`BaseSom`'s per-epoch recomputation:

```
H = exp(-dm**2 / (2 * sigma**2))          # (K, K)
```

Per iteration:

1. **BMU assignment.** Reuse the existing kernel:
   `_, winners = numba_find_winners_euclidean(X, weights)`.

2. **Per-(winner, class) sums.** Flatten winner/class into one index:
   ```
   idx = winners * n_classes + y_idx                  # (n_samples,)
   S_flat = zeros((K * C, D)); np.add.at(S_flat, idx, X)
   n_flat = np.bincount(idx, minlength=K * C)
   S_flat = S_flat.reshape(K, C, D)
   n_flat = n_flat.reshape(K, C)
   ```

3. **Majority-vote node labels**, recomputed every iteration (Kohonen's
   Comment 2 — dynamic relabeling can improve accuracy over fixed labels):
   ```
   node_labels = argmax(n_flat, axis=1)                # (K,)
   ```

4. **Neighborhood-weighted aggregation:**
   ```
   T  = (H @ S_flat.reshape(K, C * D)).reshape(K, C, D)
   Tn = H @ n_flat                                      # (K, C)
   ```

5. **Signed numerator/denominator.** Derived from expanding the LVQ sign
   rule `s(t) = +1` if sample and target-neuron class match, else `-1`,
   summed over the neighborhood (see chat derivation predating this spec):
   ```
   numerator_i   = 2 * T[i, label_i]  - T[i].sum(axis=0)
   denominator_i = 2 * Tn[i, label_i] - Tn[i].sum()
   ```
   Note `T[i].sum(axis=0) == (H @ S_c)[i]` where `S_c` is the ordinary
   (class-agnostic) per-winner sum — so no separate unsigned aggregation is
   needed; it falls out of the same `T`/`Tn` tensors.

6. **Update with stability guard** (Kohonen's Comment 1): only update a
   neuron if its denominator is positive.
   ```
   mask = denominator > 0
   weights[mask] = numerator[mask] / denominator[mask, None]
   ```

After `n_iter` iterations, `weights_` and `node_labels_` (mapped back to
original class labels via `classes_`) are fixed as fitted attributes.

`predict(X)` reuses `numba_find_winners_euclidean(X, self.weights_)` for BMU
search, then looks up `self.node_labels_[winners]`.

### Edge cases (deliberately unhandled — YAGNI)

- **Dead neurons** (never a BMU across all iterations): majority vote on an
  all-zero row returns class index 0 by `argmax` default. Weight is never
  updated regardless (denominator ≤ 0), so this only matters if such a
  neuron later becomes BMU for unseen test data. No nearest-neighbor label
  fallback (unlike `SomVQ._fit`'s dead-neuron handling) — add only if this
  proves to be a real problem.
- **No convergence check** — fixed `n_iter`, no early stopping.

### Testing plan

- Unit test: small synthetic 2-cluster case. Assert LVQ refinement does not
  worsen classification error versus plain `SomVQ` nearest-neighbor
  labeling.
- Smoke test against a real dataset, following `tests/test_som_base.py`
  conventions, compatible with `pytest -m "not slow"`.
- Determinism test: identical input to `fit` produces identical
  `weights_`/`node_labels_` across runs (no RNG in the algorithm).

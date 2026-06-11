# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repo.

## Commands

```bash
# Install dev dependencies
uv sync --group dev

# Run tests (fast only — skip slow regression)
uv run pytest -m "not slow"

# Run all tests including slow regression
uv run pytest

# Run a single test file
uv run pytest tests/test_som_base.py

# Run a single test by name
uv run pytest tests/test_som_base.py::test_cosine_metric_smoke

# Lint / format
uv run ruff format .
uv run ruff check .

# Type check
uv run ty check dbgsom/

# Run a benchmark
uv run python benchmarks/cutoff_benchmark.py
```

## Architecture

### Package layout

```
dbgsom/
  BaseSom.py       — all core logic (training loop, growth, BMU search, metrics)
  SomVQ.py         — unsupervised clustering subclass (ClusterMixin + TransformerMixin)
  SomClassifier.py — supervised classification subclass (ClassifierMixin)
  _kernels.py      — Numba JIT kernels (BMU search variants, Voronoi centers, decay)
  KDisj.py         — standalone SOM for binary disjunctive tables (separate from BaseSom)
```

Public API: `from dbgsom import SomVQ, SomClassifier`.

### Training flow (`BaseSom`)

1. `fit()` → `_initialize_som()` → `_grow_som()`
2. Two phases via `coarse_training_frac` (default 0.5):
   - **Coarse**: σ decays from `sigma_start` to `sigma_end`; neurons grow where error > GT
   - **Fine**: σ fixed at `sigma_fine`; no growth; stops at convergence
3. Per epoch: BMU search → weight update → error accumulation → growth check

### Key internal state

| Attribute          | Type                         | What it stores                                                                    |
| ------------------ | ---------------------------- | --------------------------------------------------------------------------------- |
| `som_`             | `nx.Graph`                   | Nodes = neurons (with `weight`, `error`, `epoch_created`); edges = graph topology |
| `neurons_`         | `list[tuple]`                | Ordered list of `(row, col)` node keys; index matches `weights_` rows             |
| `_node_to_idx`     | `dict[tuple, int]`           | O(1) lookup: node key → index in `neurons_`                                       |
| `weights_`         | `ndarray (K, D)`             | Extracted from graph each time neurons are added; source of truth is `som_`       |
| `_distance_matrix` | `ndarray (K, K) int16`       | All-pairs shortest-path hop counts on `som_` graph                                |
| `_neighbor_matrix` | `ndarray (K, max_deg) int64` | Padded 1-hop adjacency for pointer search                                         |

### Growth mechanism

- **Growing threshold**: `GT = lambda_ * ||std(X)||` (Qu et al. 2019)
- Neurons with accumulated error > GT → "boundary nodes"
- New neurons inserted at free grid positions adjacent to boundary nodes; weight init by reflecting opposite neighbor through boundary
- After insertion: `_node_to_idx` and `neurons_` updated immediately; `_distance_matrix` extended incrementally in O(K²); `weights_` and `_neighbor_matrix` rebuilt at next epoch start

### BMU search dispatch (`_get_winning_neurons`)

| Condition                                      | Kernel used                                                                  |
| ---------------------------------------------- | ---------------------------------------------------------------------------- |
| First epoch or `pointer_search="none"`         | `numba_find_winners_euclidean` / `_cosine` (full scan)                       |
| `pointer_search="fine"` (default) + fine phase | `numba_find_winners_pointer` / `_cosine` (graph hill-climb from prev winner) |
| `pointer_search="all"`                         | pointer from epoch 2 onward                                                  |
| n_bmu > 1 (topographic error post-training)    | `euclidean_distances` + argpartition                                         |

### Distance matrix

`_distance_matrix` stores **graph hop counts** as `int16` (not weight-space Euclidean distances). Callers needing float (e.g. topographic product, kernel) cast locally with `.astype(np.float64)`. Init via `scipy.sparse.csgraph.shortest_path`; extended incrementally via `_extend_distance_matrix()` — correct only because edges never removed.

### Neighborhood kernel dispatch (`_calculate_gaussian_neighborhood`)

Returns dense `ndarray` or `csr_array` by sparsity:

- `cutgauss`: mask `dm <= neighborhood_cutoff * σ`; sparse path when >90% zeros
- `gaussian`: threshold at h < 1e-6; same sparsity check
- `cutgauss_phase` (default `"fine"`) auto-switches to cutgauss in fine phase regardless of `neighborhood_function`; at σ≤1 and K≥200 this yields ~98% sparsity and activates CSR where gaussian stays dense

Weight update in `_update_weights()` dispatches on `issparse()`.

### Accuracy vs. performance paths

All heuristic shortcuts are phase-gated or opt-in. The defaults are the **performance path** — empirically validated to match full-accuracy quality while being significantly faster.

| Parameter | Accuracy (slow) | Performance (default) | Mechanism |
| ------------------- | --------------- | --------------------- | ------------------------------------------- |
| `pointer_search` | `"none"` | `"fine"` | O(N·K) full scan vs. O(N·deg) graph walk |
| `cutgauss_phase` | `"none"` | `"fine"` | Dense Gaussian vs. sparse CSR in fine phase |
| `neighborhood_function` | `"gaussian"` | `"gaussian"` | Full Gaussian in coarse regardless |

Coarse phase always uses full Gaussian + full BMU scan (even with defaults): topology formation is correctness-critical. Both shortcuts activate only in the fine phase, where the map is stable and σ is small.

### sklearn compatibility

`SomVQ` and `SomClassifier` pass `check_estimator`. Param validation uses `_parameter_constraints` (sklearn `Interval` / `StrOptions`). All hyperparams in `__init__` match `self.param_name` exactly — required for `clone()` and `get_params()`.

## Important constraints

- `_distance_matrix` is `int16`: never write `np.inf`. Cast to float64 first.
- `neurons_` order must stay in sync with `_node_to_idx` and `weights_` rows. Both updated together in `_add_node_to_graph()`.
- Incremental distance matrix update correct only while `_add_new_connections()` adds edges, never removes.
- Numba kernels in `_kernels.py` are `cache=True` — first call compiles, subsequent load from `__pycache__`. JIT signature change requires cache clear.
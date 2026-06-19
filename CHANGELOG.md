# Changelog

## v1.3.0 — 2026-06-07

### Breaking Changes

- Removed `pointer_search_radius` parameter — radius hardcoded to 1 (no quality gain at r>1 in fine phase)
- Removed `color='pca_rgb'` plot option — redundant with `layout='pca'`
- Removed unused `n_jobs` parameter (was stored but never read)
- `pandas` and `seaborn` are now an optional extra — only needed for `plot()`:
  `pip install dbgsom[viz]`
- Removed broken `labels` parameter from `KDisj.plot()`

### New Features

- `cutgauss_phase` parameter (default `"fine"`) — auto-switches to sparse cutgauss kernel in fine phase regardless of `neighborhood_function`
- `neighborhood_cutoff` parameter (default `3.0`) — cutgauss truncation radius, tuned to ~1% kernel value vs. previous hardcoded 2σ (13.5%)
- Pre-growth weight smoothing (Kohonen eq. 3.80): `smoothing_steps` (default `0`), `smoothing_epsilon` (default `0.5`) — improves topographic error on insertion
- Pointer-based BMU search extended to cosine metric (previously Euclidean-only)
- Map growth no longer triggers a full BMU re-scan — reuses prior winners for pointer search
- `plot()` raises `ValueError` on unknown `color` value instead of silently rendering uncoloured

### Bug Fixes

- `sigma_start` default decoupled from sample count — was scaling with `effective_max_neurons`, causing oscillation that prevented growth past 4 neurons on large low-dimensional datasets (e.g. image pixels); now fixed at `1.0`
- `transform()`: replaced `SparseCoder` with `scipy.optimize.nnls` — prior normalization stripped brightness/magnitude before sparse coding, mixing same-hue different-brightness samples
- Cosine metric: restored unit-norm invariant on neuron weights after smoothing and growth (was biasing BMU distances for up to one epoch per growth event)
- Fixed `NameError` when `pointer_search` enabled with `metric="cosine"` (missing `numba_find_winners_pointer_cosine`)

### Performance

- Replaced Floyd-Warshall full recompute (`O(K⁴)` total) with incremental `O(K²)` distance-matrix extension per growth event; `_distance_matrix` now `int16` (8× memory reduction)
- Sparse CSR kernel for neighborhood + weight update activates only when sparsity > 90% (small σ / fine phase); dense BLAS stays default in coarse phase
- `_build_neighbor_matrix` reads 1-hop adjacency directly from graph instead of scanning the distance matrix (~3× faster at growth time)

### Dependencies

- Added explicit `scipy>=1.10.0` dependency (previously transitive via sklearn)

### Documentation

- `CLAUDE.md` / docstrings: documented accuracy vs. performance code paths, corrected complexity analysis (Floyd-Warshall references → incremental `O(K³)`)
- Restructured JOSS paper statement of need, complexity, and growth-stability sections; fixed bibliography and LaTeX escape issues
- Added Sphinx-Gallery image color quantization and chess clustering example notebooks

---

## v1.2.5 — 2026-06-07

### New Features

- **Fold-angle topology analysis** — new metric for evaluating map topology quality
- **TE adjacency fix** — corrected Topographic Error adjacency computation

### Documentation

- README: rewrote hook text, updated `spreading_factor` → `lambda_` references
- README: embedded comparison plots with captions
- Algorithm docs: documented GT equilibrium neuron count estimate

### Community

- Added `CONTRIBUTING.md`
- Added `CODE_OF_CONDUCT.md`

### Compliance

- Removed license from README for PEP 639 compliance

---

## v1.2.4 — 2026-06-05

### Breaking Changes

- Removed `spreading_factor` and `threshold_method` → replaced by `lambda_`
- Removed mini-batch training path; full-batch only

### New Features

- `topographic_product_` attribute available after `fit()`
- Winner-stability convergence trigger
- Pointer-based BMU search (~3× speedup in fine phase)
- `SomVQ` and `SomClassifier` importable directly from package root

### Documentation

- Added Sphinx-Gallery examples
- Restructured algorithm documentation

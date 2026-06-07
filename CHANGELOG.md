# Changelog

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

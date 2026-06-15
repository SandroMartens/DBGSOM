---
title: "DBGSOM: A scikit-learn Compatible Python Implementation of the Directed Batch Growing Self-Organizing Map"
tags:
  - Python
  - machine learning
  - self-organizing map
  - clustering
  - unsupervised learning
  - vector quantization
  - topology preservation
authors:
  - name: Sandro Martens
    orcid: 0009-0005-6546-9015
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 2025-01-01 # TODO: Einreichungsdatum anpassen
bibliography: paper.bib
---

# Summary

Self-Organizing Maps (SOMs) [@Kohonen2001; @Kohonen2013] are unsupervised neural networks that learn a topology-preserving, low-dimensional representation of high-dimensional input data. The network maps input samples onto a discrete grid of prototype neurons such that similar inputs activate spatially proximate neurons. SOMs can be used for classification, clustering, vector quantization and nonlinear projection.

<!-- Classical SOMs require the grid dimensions to be specified prior to training, which in practice demands domain knowledge or trial-and-error tuning. -->

DBGSOM is a Python implementation of the Directed Batch Growing Self-Organizing Map [@Vasighi2017]. Starting from four neurons, the map grows autonomously by inserting new neurons at boundary positions where local quantization error exceeds a configurable threshold. Training follows the batch learning rule: weight updates are computed over the entire dataset per epoch, yielding faster convergence than online SOMs and eliminating the need to specify the map size in advance.

The library provides two estimators: `SomVQ` for unsupervised vector quantization and clustering, and `SomClassifier` for supervised classification, that integrate directly into standard machine learning workflows with scikit-learn. Performance critical paths are jit-compiler optimized.

# Statement of Need

scikit-learn [@Pedregosa2012] is one of the most used Python libraries for non-deep-learning machine learning. This is because it allows end-to-end processing from pre-processing, normalization, training to scoring and visualizing many different estimators.

The core library of scikit-learn doesn't contain any SOM implementation. Existing SOM libraries implement scikit-learn _inspired_ APIs, but don't follow the strict API standard [@sklearn2026]. This means they break when used together with other parts of scikit-learn that rely on specific behaviour. DBGSOM is the only library that we are aware of that fully integrates with scikit-learn. This includes `fit`, `predict`, `fit_predict`, `transform`, and `predict_proba` and enables drop-in use in cross-validation pipelines, `Pipeline` objects, and `GridSearchCV`.

`DBGSOM` addresses one of the major drawbacks of classical SOMs: The need to specify the layout and size of the map before the training. A single sensitivity parameter (`lambda`) lets the map grow until the desired accuracy is met. A convergence check can let the training stop at any time before `n_iter` if the map converged, making the algorihtm less sensitive the the a priori runtime setting.

The `transform` method departs from conventional SOM practice: rather than returning the index of the best-matching unit, it computes a sparse non-negative linear combination of prototype weights, yielding a meaningful embedding of each sample in prototype space [@Kohonen2007]. This allows a better encoding than the direct n-to-1 mapping to a single winner neuron. This representation is compatible with downstream scikit-learn estimators and dimensionality reduction workflows.

DBGSOM implements a number of changes to the textbook algorithm, that massively improve the speed of computation and allow scaling to larger datasets and larger networks.

The intended audience for DBGSOM is machine learning researchers working with SOMs and general data science practiciners who use the scikit-learn ecosystem.

# State of the field

Several Python SOM libraries exist, most notably MiniSom [@Vettigli2018], torchsom [@Berthier2025] and SuSi [@Riese2025]. The most used package, MiniSom, implements its own custom API. SuSi and torchsom implement parts of the scikit-learn API (namely some public functions), but don't follow the exact definitions. MiniSom and SuSi rely on pure Python and Numpy, while torchsom also supports GPU acceleration with CUDA.

| Library  | API                       | GPU            | Framework     | Docs              |
| -------- | ------------------------- | -------------- | ------------- | ----------------- |
| DBGSOM   | **sklearn-compatible**    | No             | Numpy + Numba | **Comprehensive** |
| MiniSom  | Custom (`train`/`winner`) | No             | Numpy         | Notebooks only    |
| SuSi     | sklearn-style             | No             | Numpy         | **Comprehensive** |
| torchsom | sklearn-style             | **Yes (CUDA)** | **PyTorch**   | **Comprehensive** |

All three implement fixed-grid SOMs that require the user to specify the grid dimensions before training. Selecting an appropriate grid size is non-trivial: too small a grid underfits the data; too large a grid wastes capacity and produces uninformative prototypes. In practice, users typically run multiple configurations and evaluate clustering metrics post-hoc.

Since any growing SOM has a dynamically changing grid, it cannot easiely be implemented into an existing library without rewriting much of the core logic.

# Software design

The core feature of the DBGSOM algorithm is the threshold parameter which defines how many neurons are added. The growing threshold `GT` is defined as: $GT = \lambda \cdot \lVert \text{std}(X) \rVert$

The DBGSOM training procedure is as follows:

1. **Initialization.** Four neurons are initialized with weights sampled from the input data. Their respective indices are arranged on a rectangular grid so that they form a square.
2. **Coarse Phase**: Multiple cycles of learning and growing.
   1. **Assignment.** Each training sample is assigned to its nearest neuron (Best Matching Unit, BMU) by Euclidean distance or Cosine distance.
   2. **Weight update.** Neuron weights are updated toward the mean of the samples assigned to them. A neighorhood function with dynamic bandwidth $\sigma$ lets neurons influence their grid neighbors weight update.
   3. **Growth.** Boundary neurons whose accumulated quantization error exceeds the growing threshold ($Qe_i > GT$) spawn new neighboring neurons.
   4. **Termination.** The Coarse Phase ends after a given number of epochs or if the map converged and no new neurons were added.
3. **Fine Phase.** Same as Coarse Phase, only that no new neurons are added and the neighborhood radius $\sigma$ stays constant. Training ends when `n_iter` epochs are completed or the map converged.

Convergence criterium is the Frobenius norm of the change of weights between epochs: $\|W_t - W_{(t-1)}\|_F < \varepsilon$, where $\varepsilon$ is set before training. The neighborhood width $\sigma$ decays over training epochs, transitioning the map from global to local organization.

Topology preservation is measured by the topographic error `Te` or topographic function `Tf`[@Villmann1997]. `Te` is defined as the proportion of samples for which the first and second BMU are not on adjacent edges on the map grid. The `Tf` measures folds and tears by computing how close or far neuron pairs are in the feature space.

`DBGSOM` is implemented in Python and uses NumPy [@Harris2020] for array operations and Numba [@Lam2015] for JIT-compiled distance computations. The map topology is represented as a NetworkX [@Hagberg2008] graph, which simplifies the implementation of neighborhood queries and the growth mechanism. Visualization is provided via seaborn objects [@Waskom2021], supporting continuous and categorical color encoding of prototype attributes.

Performance optimizations include JIT compilation of distance computations via Numba, sparse matrix multiplications for neighborhood weight updates, and a pointer based search algorithm for BMU lookup.

The package is distributed via PyPI (`pip install dbgsom`) and versioned according to semantic versioning. Continuous integration is configured via GitHub Actions, including unit tests, code quality checks with Ruff, and automated PyPI releases.

# Research impact statement

Benchmarks comparing DBGSOM to MiniSom, SuSi, KMeans, and AgglomerativeClustering are provided in the repository as Jupyter notebooks (`examples/som_comparison.ipynb`, `examples/clustering_comparison.ipynb`, `examples/manifold_comparison.ipynb`). Evaluations use the scikit-learn Digits dataset (1797 samples, 64 features, 10 classes) and the Fashion-MNIST dataset [@Xiao2017].

## Quality Metrics

On digits (1800 samples, 10 classes, 64 dimensions) with automatically determined cluster count (via DBGSOM's growing mechanism, applied as cluster count for all algorithms):

| Algorithm | n_prototypes | QE       | TE         | ARI     |
| --------- | ------------ | -------- | ---------- | ------- |
| DBGSOM    | 121          | **4.84** | **0.0257** | 0.17    |
| MiniSom   | 121          | **4.84** | 0.1833     | 0.17    |
| SuSi      | 121          | 5.69     | 0.0861     | 0.25    |
| torchsom  | 121          | 5.95     | 0.0889     | **0.5** |
| KMeans    | 121          | 4.3      | —          | 0.17    |

Kmeans is included to give a lower bound for `Qe`.

## Visualization

|                           Grid projection                            |                           PCA projection                           |
| :------------------------------------------------------------------: | :----------------------------------------------------------------: |
| ![Grid projection](paper_benchmarks/results/som_grid.png){width=80%} | ![PCA projection](paper_benchmarks/results/som_pca.png){width=80%} |

_Figure 1: DBGSOM neuron layout on the Digits dataset. Left: neurons positioned on the 2D grid; right: neuron weights projected to PCA space. Node color indicates the majority digit class; node size indicates hit count._

## Performance metrics

On a syntetic dataset with 1k samples to 90k samples, DBGSOM performes faster than the reference libraries with better quantization error.
![Training time vs. dataset size N for all compared algorithms (log-log scale). DBGSOM fast path uses pointer search and sparse neighborhood. [^1]](paper_benchmarks/results/scaling.png){width=80%}

# AI usage disclosure

No generative AI was used prior to release v1.2.0. Claude Code was used in Code: to create benchmarks, refactor code, improve performance, implement mathematical formulas, debugging. In documentation: Mainly for editing and keeping consistency between reference papers, documentation and actual implementation.

# References

[^1]: Hardware: CPU: AMD Ryzen 3700X, 8/16 cores, GPU: Nvidia RTX 5060Ti, 16 GB RAM.

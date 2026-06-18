---
title: "dbgsom: A `scikit-learn` Compatible Python Implementation of the Directed Batch Growing Self-Organizing Map"
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
date: 2025-16-06 # TODO: Einreichungsdatum anpassen
bibliography: paper.bib
---

# Summary

Self-Organizing Maps (SOMs) [@Kohonen2001; @Kohonen2013] are unsupervised neural networks that learn a topology-preserving, low-dimensional representation of high-dimensional input data. The network maps input samples onto a discrete grid of prototype neurons such that similar inputs activate spatially proximate neurons. SOMs can be used for classification, clustering, vector quantization and nonlinear projection.

`dbgsom` is a Python implementation of the Directed Batch Growing Self-Organizing Map [@Vasighi2017]. Starting from four neurons, the map grows autonomously by inserting new neurons according to a growing rule. Training follows the batch learning rule: weight updates are computed over the entire dataset per epoch, yielding faster convergence than online SOMs and eliminating the need to specify the map size in advance.

The library provides two estimators: `SomVQ` for unsupervised vector quantization and clustering, and `SomClassifier` for supervised classification, that integrate directly into standard machine learning workflows with `scikit-learn`.

Performance critical paths are jit-compiler optimized. We show that the quality of the resulting map and performance are comparable or better than similar SOM libraries.

# Statement of Need

**1. Full `scikit-learn` compatibility.**
`scikit-learn` [@Pedregosa2012] is the dominant Python library for non-deep-learning machine learning, yet its core does not include any SOM implementation. Existing SOM libraries implement `scikit-learn` _inspired_ APIs but do not conform to the strict API standard [@sklearn2026], breaking when composed with other `scikit-learn` components (pipelines, cross-validation, grid search). `dbgsom` is the only SOM library that passes `check_estimator` and integrates seamlessly into standard `scikit-learn` workflows.

**2. Automatic map size determination.**
Classical SOMs require the user to specify grid dimensions before training. Selecting an appropriate size is non-trivial: too small a grid underfits; too large a grid wastes capacity and produces uninformative prototypes. In practice this forces practitioners to train multiple configurations and evaluate clustering metrics post-hoc. `dbgsom` removes this burden: starting from four neurons, the map grows autonomously until the data structure is captured, guided by a principled quantization-error threshold.

**3. Target audience.**
The intended audience is machine learning researchers working with SOMs and data science practitioners using the `scikit-learn` ecosystem who need a drop-in, topology-learning estimator without manual grid tuning.

# State of the field

Several Python SOM libraries exist, most notably `MiniSom` [@Vettigli2018], `torchsom` [@Berthier2025; @Berthier2025a] and `SuSi` [@Riese2025]. There exist some GSOM [@Alahakoon2000] packages: `pygsom` [@thimalk2026] and `GSOM` [@Sales2020].

The most used package, `MiniSom`, implements its own custom API. `SuSi` and `torchsom` implement parts of the `scikit-learn` API (some public functions like `fit` and `predict`), but don't follow the exact definitions. `MiniSom` and `SuSi` rely on pure Python and Numpy, while `torchsom` also supports GPU acceleration with CUDA.

Both GSOM packages were not included because of the lack of documentation, tests, recent updates and non-standard API.

| Library  | API                       | GPU            | Framework     | Docs              |
| -------- | ------------------------- | -------------- | ------------- | ----------------- |
| `dbgsom` | **sklearn-compatible**    | No             | Numpy + Numba | **Comprehensive** |
| MiniSom  | Custom (`train`/`winner`) | No             | Numpy         | Notebooks only    |
| SuSi     | sklearn-style             | No             | Numpy         | **Comprehensive** |
| torchsom | sklearn-style             | **Yes (CUDA)** | **PyTorch**   | **Comprehensive** |

All three implement fixed-grid SOMs that require the user to specify the grid dimensions before training.

Any growing SOM has a dynamically changing grid. Therefore it cannot easiely be implemented into an existing library that uses a static grid without rewriting much of the core logic. DBGSOM especially doesn't have a rectangular layout that can be represented as a two dimensional standard array.

# Software design

`dbgsom` is implemented in Python and uses NumPy [@Harris2020] for array operations and Numba [@Lam2015] for JIT-compiled distance computations. Sparse matrix operations are performed using SciPy [@Virtanen2020]. The map topology is represented as a NetworkX [@Hagberg2008] graph. Visualization is provided via seaborn [@Waskom2021]. General API behaviour like input validation, error messages, output formats etc. are either directly interhited from `scikit-learn` or are tested against `scikit-learn` standards.

Numpy is the default library for linear algebra and array operations in Pyton. Numba allows developers to speed up Python, and specifically Numpy, code by just-in-time compilation. It needs minimal change of the original code and only a small warm up time at program start. NetworkX as graph backend simplifies the implementation of neighborhood queries and the growth mechanism that happen in a growing SOM. Seaborn supports continuous and categorical color encoding of prototype attributes, making it well suited for graph visualizations. All dependencies integrate well with each other.

The core feature of the `dbgsom` algorithm is the threshold parameter which defines how many neurons are added. The growing threshold `GT` is defined as: $GT = \lambda \cdot \lVert \text{std}(X) \rVert$

The `dbgsom` training procedure is as follows:

1. **Initialization.** Four neurons are initialized with weights sampled from the input data. Their respective indices are arranged on a rectangular grid so that they form a square.
2. **Coarse Phase**: Multiple cycles of learning and growing.
   1. **Assignment.** Each training sample is assigned to its nearest neuron (Best Matching Unit, BMU) by Euclidean distance or Cosine distance.
   2. **Weight update.** Neuron weights are updated toward the mean of the samples assigned to them. A neighorhood function with dynamic bandwidth $\sigma$ lets neurons influence their grid neighbors weight update.
   3. **Growth.** Boundary neurons whose accumulated quantization error exceeds the growing threshold ($Qe_i > GT$) spawn new neighboring neurons.
   4. **Termination.** The Coarse Phase ends after a given number of epochs or if the map converged and no new neurons were added.
3. **Fine Phase.** Same as Coarse Phase, only that no new neurons are added and the neighborhood radius $\sigma$ stays constant. Training ends when `n_iter` epochs are completed or the map converged.

Convergence criterium is change of weights between epochs. The neighborhood width $\sigma$ decays over training epochs, transitioning the map from global to local organization.

Topology preservation is measured by the topographic error `Te` or topographic function `Tf`[@Villmann1997].

The `transform` method departs from conventional SOM practice: rather than returning the index of the best-matching unit, it computes a sparse non-negative linear combination of prototype weights, yielding a meaningful embedding of each sample in prototype space [@Kohonen2007]. This allows a better encoding than the direct n-to-1 mapping to a single winner neuron.

`dbgsom` implements a number of changes to the textbook algorithm, that improve the speed of computation and allow scaling to larger datasets and larger networks. Performance optimizations include JIT compilation of distance computations via Numba, sparse matrix multiplications for neighborhood weight updates, and a pointer based search algorithm for BMU lookup.

The package is distributed via PyPI (`pip install dbgsom`) and versioned according to semantic versioning. Continuous integration is configured via GitHub Actions, including unit tests, code quality checks with Ruff, and automated PyPI releases.

# Research impact statement

`dbgsom` is used as the SOM backend for the `dsl2som` clustering library by this author [@Martens2026].

Benchmarks comparing `dbgsom` to `MiniSom`, `SuSi`, `KMeans`, and `AgglomerativeClustering` are provided in the repository as Jupyter notebooks (`examples/som_comparison.ipynb`, `examples/clustering_comparison.ipynb`, `examples/manifold_comparison.ipynb`). Evaluations use the `scikit-learn` Digits dataset (1797 samples, 64 features, 10 classes) and the Fashion-MNIST dataset [@Xiao2017].

**Quality Metrics**. On digits with automatically determined cluster count (via `dbgsom`'s growing mechanism, applied as cluster count for all algorithms):

| Algorithm  | Prototypes | Quantization error | Topographic error | Adjusted Rand index |
| ---------- | ---------- | ------------------ | ----------------- | ------------------- |
| `dbgsom`   | 127        | **4.99**           | **0.03**          | 0.18                |
| `MiniSom`  | 132        | **4.99**           | 0.14              | 0.16                |
| `SuSi`     | 132        | 5.79               | 0.07              | **0.21**            |
| `torchsom` | 132        | 5.12               | 0.09              | 0.16                |
| `KMeans`   | 127        | 4.38               | —                 | 0.17                |

Kmeans is included to give a lower bound for `Qe`.

**Performance metrics**. On a synthetic dataset with 1k to 90k samples (AMD Ryzen 3700X, 16 GB RAM, Nvidia RTX 5060Ti), `dbgsom` performs faster than the reference libraries.

![Training time vs. dataset size N for all compared algorithms (log-log scale). `dbgsom` fast path uses pointer search and sparse neighborhood.](paper_benchmarks/results/scaling.png){width=80%}

**Visualization**. `dbgsom` provides standard visualization capabilities for SOMs. Nodes can be plotted using grid coordinates or by PCA projection of the original dataset. Node sizes and colors can encode different properties of each neuron.

![`dbgsom` neuron layout on the Digits dataset: neurons positioned on the 2D grid. Node color indicates the majority digit class; node size indicates hit count.](paper_benchmarks/results/som_grid.png){width=60%}

![`dbgsom` neuron layout on the Digits dataset: neuron weights projected to PCA space. Node color indicates the majority digit class; node size indicates hit count.](paper_benchmarks/results/som_pca.png){width=60%}

# AI usage disclosure

No generative AI was used prior to release v1.2.0. Claude Code was used in Code: to create benchmarks, refactor code, improve performance, implement mathematical formulas, debugging. In documentation: Mainly for editing and keeping consistency between reference papers, documentation and actual implementation.

All documentations, implementations and experimental results were verified to the best of the authors knowledge. Experiments can be reproduced in the `paper_benchmarks` folder.

# References

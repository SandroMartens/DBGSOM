![license](https://img.shields.io/github/license/SandroMartens/DBGSOM)
![readthedocs](https://img.shields.io/readthedocs/dbgsom)
[![DOI](https://zenodo.org/badge/454955249.svg)](https://doi.org/10.5281/zenodo.20525611)
[![Python package](https://github.com/SandroMartens/DBGSOM/actions/workflows/python-package.yml/badge.svg)](https://github.com/SandroMartens/DBGSOM/actions/workflows/python-package.yml)
[![Publish to PyPI](https://github.com/SandroMartens/DBGSOM/actions/workflows/publish.yml/badge.svg)](https://github.com/SandroMartens/DBGSOM/actions/workflows/publish.yml)
[![CodeQL Advanced](https://github.com/SandroMartens/DBGSOM/actions/workflows/codeql.yml/badge.svg)](https://github.com/SandroMartens/DBGSOM/actions/workflows/codeql.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# DBGSOM

DBGSOM is short for _Directed Batch Growing Self-Organizing Map_. A SOM is a type of artificial neural network that is used to produce a low-dimensional representation of a higher-dimensional data set while preserving the topological structure of the data. It can be used for supervised and unsupervised vector quantization, classification and many different data visualization tasks.

## Features

- Compatible with scikit-learn's API — drop-in replacement for other clustering and classification algorithms
- Can handle high-dimensional and non-uniform data distributions
- Good results without extensive parameter tuning
- Better topology preservation and faster training time than classical SOMs
- Interpretability of the results through interactive plotting

## How it works

The DBGSOM algorithm builds a two-dimensional map of prototypes (neurons) where each neuron is connected to its neighbors. Four neurons are initialized with random weight vectors drawn from the input data. During training every sample is assigned to its nearest neuron (best matching unit), and the neuron weights are updated towards the samples mapped to them. Neighboring neurons influence each other's updates so that the low-dimensional ordering of the map is preserved. A growing mechanism expands the map as needed: new neurons are inserted at boundary positions where the quantization error exceeds a configurable growing threshold.

## How to install

### Download from PyPI

DBGSOM can be installed from PyPI via `uv` (recommended):

```bash
uv add dbgsom
```

or with pip:

```bash
pip install dbgsom
```

### Install from source

Clone the repository and install with `uv` (recommended):

```bash
git clone https://github.com/SandroMartens/DBGSOM.git
cd DBGSOM
uv sync
```

Alternatively with `pip`:

```bash
git clone https://github.com/SandroMartens/DBGSOM.git
cd DBGSOM
pip install -e .
```

## Usage

DBGSOM implements the scikit-learn API and provides two estimators:

| Class           | Use case                                      |
| --------------- | --------------------------------------------- |
| `SomVQ`         | Unsupervised clustering / vector quantization |
| `SomClassifier` | Supervised classification                     |

### Clustering / Vector Quantization

```python
from dbgsom import SomVQ
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)

vq = SomVQ(spreading_factor=0.5, max_neurons=80)
labels = vq.fit_predict(X)

print(f"Neurons: {len(vq.neurons_)}")
print(f"Quantization error: {vq.quantization_error_:.4f}")
print(f"Topographic error:  {vq.topographic_error_:.4f}")
```

### Classification

```python
from dbgsom import SomClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf = SomClassifier(spreading_factor=0.5, max_neurons=80)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))           # accuracy
proba = clf.predict_proba(X_test)          # class probabilities
```

### Transform

Both estimators implement `transform()`, which represents each sample as a sparse non-negative linear combination of the prototype weight vectors:

```python
coefs = vq.transform(X)   # shape (n_samples, n_prototypes)
```

### Visualization

`plot()` renders the SOM neurons as dots and the neighbourhood edges as grey lines, all via seaborn objects.

```python
vq.plot(color="density")                       # continuous → colour gradient
clf.plot(color="label")                        # categorical → colour legend
vq.plot(color="hit_count", pointsize="error")  # colour + size encoding
vq.plot(color="density", layout="pca", palette="magma_r")
vq.plot(color="pca_rgb")                       # RGB colour from PCA of weight vectors
```

Supported attributes for `color` / `pointsize`:
`'label'`, `'epoch_created'`, `'error'`, `'average_distance'`, `'density'`, `'hit_count'`

| Parameter   | Options                       | Description                                                                            |
| ----------- | ----------------------------- | -------------------------------------------------------------------------------------- |
| `color`     | any node attribute            | Numeric attributes → continuous colour scale; int/str with ≤ 20 unique values → legend |
| `pointsize` | any numeric attribute         | Node size proportional to attribute value                                              |
| `layout`    | `'grid'` _(default)_, `'pca'` | Node placement algorithm                                                               |
| `palette`   | any Matplotlib colormap       | Applied to the colour mapping                                                          |

## Examples

| Example                                                         | Description                                                                                                                                                                |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![example](examples/export/2d_example.png)                      | With two-dimensional input we can clearly see how the prototypes (red) approximate the input distribution (white) while preserving the square topology to their neighbors. |
| ![The fashion mnist dataset](examples/export/fashion_mnist.png) | After training on the Fashion-MNIST dataset we can plot the weigth of each prototype. Neighboring prototypes are pairwise similar.                                         |
| ![digits](examples/export/digits_classes.png)                   | Each prototype is coloured by its majority class. Samples from the same class cluster together. Trained on MNIST digits.                                                   |
| ![darknet_pca](examples/export/darknet_pca.png)                 | Linear transformations like PCA can colour-code relative distances between prototypes in the input space. See the _darknet_ example notebook.                              |

## Dependencies

- Python ≥ 3.12
- numpy
- numba
- NetworkX
- tqdm
- scikit-learn
- scikit-image
- seaborn
- matplotlib
- pandas
- scipy

## References

- _A directed batch growing approach to enhance the topology preservation of self-organizing map_, Mahdi Vasighi and Homa Amini, 2017, <http://dx.doi.org/10.1016/j.asoc.2017.02.015>
- Reference implementation by the authors in Matlab: <https://github.com/mvasighi/DBGSOM>
- _Statistics-enhanced Direct Batch Growth Self-Organizing Mapping for efficient DoS Attack Detection_, Xiaofei Qu et al., 2019, [10.1109/ACCESS.2019.2922737](https://ieeexplore.ieee.org/document/8736234)
- _Entropy-Defined Direct Batch Growing Hierarchical Self-Organizing Mapping for Efficient Network Anomaly Detection_, Xiaofei Qu et al., 2021, 10.1109/ACCESS.2021.3064200
- _Self-Organizing Maps_, 3rd Edition, Teuvo Kohonen, 2003
- _MATLAB Implementations and Applications of the Self-Organizing Map_, Teuvo Kohonen, 2014
- _Smoothed self-organizing map for robust clustering_, P. D'Urso, L. De Giovanni and R. Massari, 2019, <https://doi.org/10.1016/j.ins.2019.06.038>

## License

dbgsom is licensed under the MIT license.

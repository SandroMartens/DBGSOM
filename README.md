# DBGSOM

DBGSOM is short for _Directed batch growing self-organizing map_. A SOM is a type of artificial neural network that is used to to produce a low-dimensional  representation of a higher dimensional data set while preserving the topological structure of the data.  It can be used for unsupervised vector quantization, classification and many different data visualization tasks.

This project was inspired by the great [MiniSom](https://github.com/JustGlowing/minisom).

## Features

- Compatible with scikit-learn's API and can be used as a drop-in replacement for other clustering and classification algorithms.
- Can handle high-dimensional and non-uniform data distributions.
- Good results without parameter tuning.
- Better topology preservation than classical SOMs.
- Interpretability of the results through plotting.

## How it works

The DBGSOM algorithm works by constructing a two-dimensional map of prototypes (neurons) where each neuron is connected to its neighbors. The first neurons on the map are initialized with random weights from the input data. The input data is then presented to the SOM. Each sample gets assigned to it's nearest neuron. The neuron weights are then updated to the samples that were mapped to each neuron. Neighboring neurons affect each others updates, so the low dimensional ordering of the map is preserved. The DBGSOM algorithm uses a growing mechanism to expand the map as needed. New neurons are added to the edge of the map where the quantization error of the boundary neurons is above a given growing threshold.

## How to install

(WiP)

```Powershell
git clone https://github.com/SandroMartens/DBGSOM.git
cd dbgsom
pip install -e .
```

## Usage

dbgsom implements the scikit-learn API.

```Python
from dbgsom import DBGSOM

quantizer = DBGSOM()
quantizer.fit(X=data_train)
labels_train = quantizer.labels_
labels_test = quantizer.predict(X=data_test)

```

## Examples

Here are a few example use cases for dbgsom (WiP)

|Example|Description|
|-|-|
|![example](examples/2d_example.png)| With a two dimensional input we can clearly see how the protoypes (red) approximate the input distribution (white) while still preserving the square topology to their neighbors.|
|![The fashion mnist dataset](examples/fashion_mnist.png) | After training the SOM on the fashion mnist dataset we can plot the nearest neighbor of each prototype. We can see that the SOM ordered the prototypes in a way that neighboring prototypes are pairwise similar. |
|![digits](examples/digits_classes.png) | We can show the majority class each prototype represents. Samples from the same class are clustered together. The SOM was train on mnist digits.|

## Dependencies

- Numpy
- NetworkX
- tqdm
- scikit-learn
- seaborn

## References

- _A directed batch growing approach to enhance the topology preservation of self-organizing map_, Mahdi Vasighi and Homa Amini, 2017, <http://dx.doi.org/10.1016/j.asoc.2017.02.015>
- Reference implementation by the authors in Matlab: <https://github.com/mvasighi/DBGSOM>
- _Statistics-enhanced Direct Batch Growth Self- organizing Mapping for efficient DoS Attack Detection_, Xiaofei Qu et al., 2019, [10.1109/ACCESS.2019.2922737](https://ieeexplore.ieee.org/document/8736234)
- _Entropy-Defined Direct Batch Growing Hierarchical Self-Organizing Mapping for Efficient Network Anomaly Detection_, Xiaofei Qu et al., 2021 10.1109/ACCESS.2021.3064200
- _Self-Organizing Maps_, 3rd Edition, Teuvo Kohonen, 2003
- _MATLAB Implementations and Applications of the Self-Organizing Map_, Teuvo Kohonen, 2014

## License

dbgsom is licensed under the MIT license.

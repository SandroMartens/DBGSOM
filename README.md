# DBGSOM

DBGSOM is short for _A directed batch growing approach to enhance the topology preservation of self-organizing map_ (SOM) (after the paper with the same title).  
This project was inspired by the great [MiniSom](https://github.com/JustGlowing/minisom).

## How it works

In a growing SOM we start with a small map and extend the grid according to some growing criterion. In case of the DBGSOM algorithm, we add neurons the edge of the map where the quantization error of the boundary neurons is above a given growing threshold.

## Usage

dbgsom implements the scikit-learn API.

```Python
from dbgsom import DBGSOM

quantizer = DBGSOM()
quantizer.fit(X=data_train)
labels = quantizer.labels_
labels_new = quantizer.predict(data_test)

```

## Dependencies

- Numpy
- NetworkX
- Scipy
- tqdm
- scikit-learn
- pynndescent

## References

- _A directed batch growing approach to enhance the topology preservation of self-organizing map_, Mahdi Vasighi and Homa Amini, 2017, <http://dx.doi.org/10.1016/j.asoc.2017.02.015>
- Reference implementation by the authors in Matlab: <https://github.com/mvasighi/DBGSOM>
- _Self-Organizing Maps_, 3rd Edition, Teuvo Kohonen, 2003
- _MATLAB Implementations and Applications of the Self-Organizing Map_, Teuvo Kohonen, 2014

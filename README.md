# DBGSOM
A directed batch growing approach to enhance the topology preservation of self-organizing map (SOM).

## How it works
In a growing SOM we start with a small map and extend the grid according to some growing criterion. In case of the DBGSOM algorithm, we add neurons the edge of the map where the boundary neuron has the greatest quantization error.

## Dependencies
- Numpy
- Pandas
- NetworkX

## References
- _A directed batch growing approach to enhance the topology preservation of self-organizing map_, Mahdi Vasighi and Homa Amini, 2017, http://dx.doi.org/10.1016/j.asoc.2017.02.015
- _Self-Organizing Maps_, 3rd Edition, Teuvo Kohonen, 2003
- _MATLAB Implementations and Applications of the Self-Organizing Map_, Teuvo Kohonen, 2014

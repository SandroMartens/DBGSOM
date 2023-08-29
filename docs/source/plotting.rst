Plotting Results
================

One of the mayor features of a SOM is the ability to project the high dimensional input data in a two dimensional output space. It shares some properties with nonlinear projection methods such as multidimensional scaling (MDS), especially the Sammon projection. However, the SOM is quantized. The model may not be a replica of any input item but only a local average over a subset of items that are most similar to it. The resolution of the projection depends on the number of neurons trained. We can use this property to calculate additional statistical information about the neurons, which can then be plotted. 

Each neuron has a 2d coordinate, which allows us to visualize the graph as a simple scatter plot. Note that this visualization does not show any distances in the input space.

We need a trained `som` object. We then call the `plot` function on that object:

.. code-block:: python
    from dbgsom.dbgsom_ import DBGSOM
    from sklearn.datasets import load_digits
    digits_X, digits_y = load_digits(return_X_y=True)

    som = DBGSOM()
    som.fit(digits_X)
    som.plot()


Additional attributes
---------

We can use the `color` and `size` parameters to enhance the basic visualization. During computation, different data are computed for each neuron, which can be coded als color or size of the scatter plot. In `DBGSOM` six attributes are currently implemented:

- `label`
    Label of the prototype when trained supervised.
- `epoch_created`
    Training epoch in which a neuron was added.
- `error`
    Quantization error or entropy, depending on the growth criterion used.
- `average_distance`
    Distance to neighboring neurons in the input space. The creates a U matrix.
- `density`
    A local density estimate around each neuron.
- `hit_count`
    Number of samples the prototype represents.

We can plot two attributes at the same time by using one value for the color and one for the size of each node. The `palette` parameter accepts all valid seaborn/matplotlib color scales.

.. code-block::python
    som.plot(color="density", size="error", palette="viridis")

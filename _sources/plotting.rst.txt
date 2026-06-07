Plotting Results
================

One of the major features of a SOM is the ability to project high-dimensional input data into a two-dimensional output space. It shares some properties with nonlinear projection methods such as multidimensional scaling (MDS). However, the SOM is quantized: each neuron represents a local average over the subset of samples most similar to it. The resolution of the projection depends on the number of neurons trained.

Each neuron has a 2D coordinate on the grid, which allows us to visualize the map as a scatter plot. Note that the default grid layout does not show distances in the input space — use ``layout='pca'`` for a layout that reflects input-space distances.

Basic Usage
-----------

.. code-block:: python

    from dbgsom import SomVQ
    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)

    som = SomVQ(spreading_factor=0.5, max_neurons=80)
    som.fit(X)
    som.plot()

Node Attributes
---------------

During training, the following attributes are computed for each neuron and can be used for colour or size encoding:

- ``label`` — majority class label (supervised training only)
- ``epoch_created`` — training epoch in which the neuron was added
- ``error`` — quantization error or entropy per neuron, depending on ``growth_criterion``
- ``average_distance`` — average distance to neighbouring neurons in input space (U-matrix)
- ``density`` — kernel density estimate around each neuron
- ``hit_count`` — number of training samples the neuron represents

Color and Size Encoding
-----------------------

Use the ``color`` and ``pointsize`` parameters to encode node attributes. Numeric attributes use a continuous colour scale; categorical attributes (≤ 20 unique values) use a legend.

.. code-block:: python

    som.plot(color="density", pointsize="error", palette="viridis")

The ``palette`` parameter accepts any Matplotlib / seaborn colormap name.

Layout Options
--------------

Two layout algorithms are available via the ``layout`` parameter:

``'grid'`` (default)
    Neurons are placed at their integer SOM grid coordinates. Preserves the topological map structure but does not reflect distances in input space.

``'pca'``
    Weight vectors are projected to 2D with PCA. Node positions reflect the principal directions of variance in feature space, giving a sense of input-space distances between prototypes.

.. code-block:: python

    som.plot(color="density", layout="pca")

Data-Aligned PCA
----------------

When using ``layout='pca'``, pass the training data ``X`` to align the PCA basis with the data variance rather than the weight variance. This is useful when the weight vectors span a subspace of the data manifold.

.. code-block:: python

    som.plot(layout="pca", X=X)

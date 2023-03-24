Plotting Results
================

One of the mayor features of a SOM is the ability to project the input data in a two dimensional space. It shares some properties with nonlinear projection methods such as multidimensional scaling (MDS), especially the Sammon projection. However, the SOM is quantized. The model may not be a replica of any input item but only a local average over a subset of items that are most similar to it. The resolution of the projection depends on the number of neurons trained. We can use this property to calculate additional statistical information about the neurons, which can then be plotted. 

In `DBGSOM` six attributes are currently implemented:

- Label
    Label of the prototype when trained supervised.
- Epoch created
    Training epoch in which a neuron was added.
- error
    Quantization error or entropy, depending on the growth criterion used.
- Average distance
    Distance to neighboring neurons in the input space. The creates a U matrix.
- density
    A local density estimate around each neuron.
- hit_count
    Number of samples the prototype represents.

We can plot two attributes at the same time by using one value for the color and one for the size of each node.

ToDo: Examples
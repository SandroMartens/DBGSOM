The Algorithm
=============

Overview
--------

The SOM algorithm performs in two steps, competition among the neurons to find the winner and adaptation of the weight vector of the winner neuron and its topological neighbors. Instead of being confined to a redetermined number of neurons, DBGGSOM offers a flexible structure and requires less number of epochs compared to the original SOM which enable the ability to learn the nonlinear manifolds in high dimensional feature space. Training of the DBGGSOM starts
from a small number of initial neurons to a larger map by adding new neurons inside the network. In the DBGSOM a batch growing approach for SOM called directed batch growing self-organizing map is used. It uses the accumulative error of the neurons on the grid to direct the growing phase in
term of position and weight initialization of new neurons. New neurons can be added from boundaries by filling one of the adjacent free positions and assigning a proper weight vector in order to improve the topographic quality of the map and help the map to learn the manifold of the data in high dimensional feature space

Batch Learning Algorithm
------------------------
For a training data, the batch learning algorithm for SOMs can be performed by presentation of an input vector :math:`x_j` to all the neurons at the same time and finds a winner
neuron which the weight vector :math:`w_c` has the minimum distance to :math:`x_j`. 

The new weights are then calculated as

.. math::

    w_i^{new} = \frac{\sum_{j=1}^{k}h_{c_{j, i}} x_j}{\sum_{j=1}^{k}h_{c_{j, i}}}

where :math:`h_{c_{j, i}}` is the gaussian neighborhood function defined as:

.. math::

    h_{c_{j, i}} = exp(- \frac{{\lvert w_i - w_{c_j} \rvert}^2}{2{\sigma}^2(t)})

where :math:`w_i` is the weight vectors of the `i` th neuron and :math:`w_{c_j}` is the weight vector of winning neuron `c` for `j` th input vector. :math:`\lvert w_i - w_{c_j} \rvert` is the distance between these two prototype on the grid. :math:`\sigma` is the width of the Gaussian function which controls the cooperation of neighbor neurons in the learning process. The value of :math:`\sigma(t)` decreases with time. This procedure can be repeated a number of times specified
by the user.

Runtime complexity
------------------
The space complexity `S` of a SOM is:

.. math::
    
    S = O(d*(m+n))

There are different ways to estimate the time complexity. A rough estimate for `T` is:

.. math::

    T = O(nmde)

where

* `n` is the number of data samples
* `m` is the number of neurons
* `d` is the data dimension and
* `e` is the number of training iterations
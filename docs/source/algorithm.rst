Algorithm
=========

DBGSOM Framework
--------------------

The SOM algorithm performs in two steps. First, competition among the neurons to find the winner and second, adaptation of the weight vector of the winner neuron and its topological neighbors. Instead of being confined to a redetermined number of neurons, DBGGSOM offers a flexible structure and requires less number of epochs compared to the original SOM which enable the ability to learn the nonlinear manifolds in high dimensional feature space.

Training of the DBGGSOM starts from a small number of initial neurons to a larger map by adding new neurons inside the network. In the DBGSOM a batch growing approach for SOM called directed batch growing self-organizing map is used. It uses the accumulative error of the neurons on the grid to direct the growing phase in term of position and weight initialization of new neurons. New neurons can be added from boundaries by filling one of the adjacent free positions and assigning a proper weight vector in order to improve the topographic quality of the map and help the map to learn the manifold of the data in high dimensional feature space.

Batch Learning Algorithm
************************

For a training data, the batch learning algorithm for SOMs can be performed by presentation of an input vector :math:`x_j` to all the neurons at the same time and finds a winner neuron which the weight vector :math:`w_c` has the minimum distance to :math:`x_j`. The new weights are then calculated as

.. math::
    w_i^{new} = \frac{\sum_{j=1}^{k}h_{c_{j, i}} x_j}{\sum_{j=1}^{k}h_{c_{j, i}}}

where :math:`h_{c_{j, i}}` is the Gaussian neighborhood function defined as:

.. math::
    h_{c_{j, i}} = \exp \left(- \frac{{\lvert w_i - w_{c_j} \rvert}^2}{2{\sigma}^2(t)}\right)

where :math:`w_i` is the weight vectors of the `i` th neuron and :math:`w_{c_j}` is the weight vector of winning neuron `c` for `j` th input vector. :math:`\lvert w_i - w_{c_j} \rvert` is the distance between these two prototype on the grid. :math:`\sigma` is the width of the Gaussian function which controls the cooperation of neighbor neurons in the learning process. The value of :math:`\sigma(t)` decreases with time. This procedure can be repeated a number of times specified
by the user.

Directed Horizontal Growth
**************************

We calculate a growing threshold `GT` as 

.. math::
    GT = -\log(sf) * d

where `sf` as the spreading factor chosen by the user and `d` is the dimensionality of the data. After each training epoch the accumulative error (:math:`E_i`) for each neuron is calculated as:

.. math::
    E_i = \sum_{p=1}^k \lvert x_p - w_i \rvert

where :math:`w_i` is the weight vector of the neuron `i` and `k` is the number of input vectors mapped on `i`. 

For each non boundary neuron :math:`n_i` where :math:`E_i > GT`, :math:`0.5 E_i` is distributed to neighboring boundary neurons. Then a new neuron is added to a free position to all boundary neurons where :math:`E_i > GT`. The position and weight of the new neuron are described in the paper.

First classification
********************

For a sample classification, each neuron :math:`n_i` gets assigned a label :math:`L_i`. That label is decided by a majority vote of the class labels `l` of all samples represented by that prototype: 

.. math::
    L_i = \mode(l_1 \ldots l_n)

Extensions
----------

There are currently three extensions to the original DBGSOM implemented: 

- Hierarchical SOM (HSOM), 
- Statistics Enhanced DBGGSOM (SE-DBGSOM) and
- Entropy Defined DBGSOM (ED-DBGSOM)

The HSOM can handle deeply bunched data samples, that cannot be distinguished by more neuron growth. A new, smaller SOM is created only for one neuron of the original SOM. The SE-DBGSOM uses the standard deviation of the input features to control the growth and fine classification, while the ED-DBGSOM uses the entropy of the classses of each prototype and the entropy of the features in each sample.

Hierarchical DBGSOM
*******************
We calculate a vertical growing threshold as :math:`VGT = 1.5 * GT`. After the horizontal growth phase is finished, for each neuron :math:`n_i` where :math:`E_i > VGT`, a new SOM is created and trained on all samples mapped to :math:`n_i`. This is done recursively.

Statistics Enhanced DBGSOM
***************************

Entropy Defined DBGSOM
**********************

The entropy of each neuron :math:`n_i` is given by:

.. math::
    E_i = \sum_{x \in s(x)} -p_i(x) * \log_2 p_i(x)

Here, subscript `i` denotes the neuron serial, :math:`p_i(x)` denotes the probability of the data vectors typed `x`, `s(x)` denotes the set of all data types. So the SOM grows in the direction where the classification of the samples is bad.

Fine Grained Classification
###########################
Currently not implemented.

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
* `e` is the number of training epochs
The Algorithm
=============

Runtime complexity
------------------
There are different ways to estimate the complexity of SOMs. A rough estimate for the complexity is:
.. math::

    T = O(nmde)

    S = O(nmd)

where
* `n` is the number of data samples
* `m` is the number of neurons
* `d` is the data dimension and
* `e` is the number of training iterations
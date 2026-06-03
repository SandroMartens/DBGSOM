Algorithm
=========

DBGSOM Framework
--------------------

The SOM algorithm performs in two steps. First, competition among the neurons to find the winner and second, adaptation of the weight vector of the winner neuron and its topological neighbors. Instead of being confined to a predetermined number of neurons, DBGSOM offers a flexible structure and requires fewer epochs compared to the original SOM, which enables the ability to learn the nonlinear manifolds in high-dimensional feature space.

Training of the DBGSOM starts from a small number of initial neurons to a larger map by adding new neurons inside the network. A batch growing approach for SOM called Directed Batch Growing Self-Organizing Map is used. It uses the accumulative error of the neurons on the grid to direct the growing phase in terms of position and weight initialization of new neurons. New neurons can be added from boundaries by filling one of the adjacent free positions and assigning a proper weight vector in order to improve the topographic quality of the map and help the map to learn the manifold of the data in high-dimensional feature space.

This design is motivated by two well-established properties: BMU search costs :math:`O(n \cdot m \cdot d)` and the weight update costs :math:`O(m^2 \cdot d)` per epoch, so a small map with few neurons converges in significantly less compute time than a large one. Growing a large map incrementally from a small initialisation is therefore both faster and less sensitive to initialisation than training a large map from scratch (Kohonen, 2014).

Batch Learning Algorithm
************************

For a training dataset, the batch learning algorithm presents all input vectors :math:`x_j` to all neurons simultaneously and finds the winner neuron whose weight vector :math:`w_c` has the minimum distance to :math:`x_j`. The new weights are then calculated as

.. math::
    w_i^{new} = \frac{\sum_{j=1}^{k}h_{c_{j, i}} \cdot s_j \cdot x_j}{\sum_{j=1}^{k}h_{c_{j, i}} \cdot s_j}

where :math:`h_{c_{j, i}}` is the neighborhood function (see below) and :math:`s_j` is a per-sample robustness weight (see `Robustness Weighting`_).

Two-Phase Training
##################

Training is divided into a **coarse phase** and a **fine phase**, controlled by ``coarse_training_frac`` (default 0.5):

- **Coarse phase** (first ``coarse_training_frac * n_iter`` epochs): the map grows. The neighborhood bandwidth :math:`\sigma` starts at ``sigma_start`` (default :math:`0.2 \cdot \sqrt{m}`, where :math:`m` is the current neuron count) and decays toward ``sigma_end`` (default :math:`0.05 \cdot \sqrt{m}`) after each growth step.
- **Fine phase** (remaining epochs): no further growth. :math:`\sigma` is fixed to ``sigma_fine`` if set, otherwise the last coarse value is kept. Small values (~0.1) concentrate updates on the BMU and minimize quantization error, as recommended by Kohonen. A SOM with a fixed (non-decaying) :math:`\sigma` is empirically known to converge in a finite number of steps (Kohonen, 2014), which guarantees termination of the fine phase.

The decay from ``sigma_start`` to ``sigma_end`` follows either an **exponential** or **linear** schedule (``decay_function`` parameter). For exponential decay the learning rate is chosen so that 99% of the drop from ``sigma_start`` to ``sigma_end`` is completed by the end of the coarse phase.

Neighborhood Functions
######################

Two neighborhood functions are available via the ``neighborhood_function`` parameter:

**Gaussian** (default, ``"gaussian"``)

.. math::
    h_{c_{j, i}} = \exp \left(- \frac{d_{ij}^2}{2{\sigma}^2}\right)

where :math:`d_{ij}` is the graph distance between neurons :math:`i` and :math:`c_j` on the SOM grid, computed via Floyd–Warshall.

**Cut Gaussian** (``"cutgauss"``)

Same as Gaussian, but set to zero for all neuron pairs with graph distance :math:`d_{ij} > 2\sigma`. This concentrates updates on a well-defined neighborhood and suppresses long-range interference.

Robustness Weighting
********************

Before the batch weight update, each sample :math:`x_j` is assigned a robustness weight

.. math::
    s_j = 1 - \left(1 - \exp\!\left(-\gamma \, \lVert x_j - w_{c_j} \rVert^2\right)\right)^{1/2}, \quad \gamma = \frac{1}{\operatorname{Var}(X)}

Samples far from their BMU (outliers) receive lower weights, making the map more robust to noise. Samples close to their BMU receive a weight near 1.

Reference: D'Urso et al., *Smoothed self-organizing map for robust clustering*, Information Sciences, 2019.

Distance Metrics
****************

Two distance metrics are supported via the ``metric`` parameter:

- ``"euclidean"`` (default): standard Euclidean distance; BMU search via Numba JIT kernel (fused distance + argmin, no n×m matrix allocated).
- ``"cosine"``: cosine dissimilarity :math:`1 - \langle x, w \rangle`; data and weights are L2-normalised before training and at each update step. BMU search via a separate Numba JIT dot-product kernel.

Directed Horizontal Growth
**************************

Growing Threshold
#################

Two methods are available for computing the growing threshold ``GT`` via ``threshold_method``:

**Statistics-Enhanced** (default, ``"se"``)

.. math::
    GT = 150 \cdot (-\log sf) \cdot \left\lVert \operatorname{std}(X) \right\rVert

where :math:`sf` is the ``spreading_factor`` and :math:`\operatorname{std}(X)` is the per-feature standard deviation of the training data. This data-adaptive formulation reflects the scale and spread of the dataset.

Reference: Qu et al., *Statistics-enhanced Direct Batch Growth Self-Organizing Mapping for efficient DoS Attack Detection*, IEEE Access, 2019.

**Classical** (``"classical"``)

.. math::
    GT = -d \cdot \log(sf)

where :math:`d` is the dimensionality of the data.

Growth Triggering
#################

Growth is **convergence-triggered**, not epoch-triggered. After each batch weight update, the change in the weight matrix is compared against ``convergence_threshold``. When the change falls below this threshold, the map is considered converged and — if still in the coarse phase and below ``max_neurons`` — a growth step is performed:

1. The accumulative error :math:`E_i` for each neuron is evaluated.
2. For each non-boundary neuron :math:`n_i` where :math:`E_i > GT`, half its error is distributed to neighboring boundary neurons.
3. New neurons are inserted at all boundary positions where :math:`E_i > GT`, starting with the highest-error neuron.

Waiting for convergence before inserting neurons is deliberate: only once the map has converged do the per-neuron error values constitute stable estimates of the true quantization error. Growing from a transient training state would make the error distribution unreliable and could direct new neurons to the wrong positions (Kohonen, 2014).

The position and weight of each new neuron are determined by the directed insertion rule (Vasighi and Amini, Section 3.3.1.1): the free adjacent position whose corner neighbor has the highest error is selected, and the new weight is initialized by reflecting the opposite neighbor through the boundary neuron. Because each inserted neuron inherits a weight derived from its already-trained neighbours, the map remains partially ordered after growth. A partially ordered map is empirically known to converge significantly faster than a randomly initialised one (Kohonen, 2001), which is why the weight initialisation rule is central to the algorithm's efficiency.

After a growth step, :math:`\sigma` is updated via the decay function and ``converged_`` is reset to ``False``, restarting the convergence check.

Growth Criterion
################

The default growth criterion is the **quantization error** of each neuron (sum of distances from mapped samples to the prototype). Alternatively, ``growth_criterion="entropy"`` uses the Shannon entropy of the class distribution per neuron, so the map grows where classification is poorest.

First Classification
********************

For sample classification, each neuron :math:`n_i` gets assigned a label :math:`L_i` as the most common class label :math:`l` of all samples represented by that prototype:

.. math::
    L_i = \operatorname{mode}(l_1, \ldots, l_n)

Extensions
----------

There are currently three extensions to the original DBGSOM implemented:

- Hierarchical SOM (HSOM)
- Statistics Enhanced DBGSOM (SE-DBGSOM)
- Entropy Defined DBGSOM (ED-DBGSOM)

The HSOM handles densely clustered data samples that cannot be distinguished by further neuron growth. A new, smaller SOM is created for neurons whose error remains high after the horizontal growth phase. The SE-DBGSOM uses the standard deviation of the input features to control the growth threshold, while the ED-DBGSOM uses the entropy of the class distribution per prototype as the growth criterion.

Hierarchical DBGSOM
*******************

After the horizontal growth phase, each neuron :math:`n_i` whose quantization error exceeds a vertical growing threshold triggers the creation of a child SOM trained on all samples mapped to :math:`n_i`. The vertical growing threshold is:

.. math::
    VGT = \tau_2 \cdot QE_0

where :math:`\tau_2` is the ``tau_2`` parameter (default 0.5) and :math:`QE_0` is the quantization error of a single-neuron SOM whose weight equals the mean of all training data. This formulation follows the GHSOM stopping criterion.

A child SOM is only created when the number of samples mapped to the neuron exceeds ``min_samples_vertical_growth``.

Reference: Qu et al., *Entropy-Defined Direct Batch Growing Hierarchical Self-Organizing Mapping for Efficient Network Anomaly Detection*, IEEE Access, 2021.

Statistics Enhanced DBGSOM
***************************

The SE-DBGSOM variant selects ``threshold_method="se"`` (the default in this implementation) and thereby computes the growing threshold depending on the standard deviation of the input features:

.. math::
    GT = 150 \cdot (-\log sf) \cdot \left\lVert \operatorname{std}(X) \right\rVert

where :math:`\operatorname{std}_i` denotes the standard deviation of all samples in the :math:`i`-th input dimension. This improved approach adapts the threshold to the scale of the dataset.

Reference: Qu et al., *Statistics-enhanced Direct Batch Growth Self-Organizing Mapping for efficient DoS Attack Detection*, IEEE Access, 2019.

Entropy Defined DBGSOM
**********************

Selected via ``growth_criterion="entropy"``. Instead of the quantization error, the Shannon entropy of the class distribution per neuron is used as the growth criterion:

.. math::
    E_i = -\sum_{x \in \mathcal{S}} p_i(x) \log_2 p_i(x)

where :math:`p_i(x)` is the fraction of samples mapped to neuron :math:`i` that carry label :math:`x`, and :math:`\mathcal{S}` is the set of all class labels. The map therefore grows in directions where classification is most uncertain.

Reference: Qu et al., *Entropy-Defined Direct Batch Growing Hierarchical Self-Organizing Mapping for Efficient Network Anomaly Detection*, IEEE Access, 2021.


Runtime complexity
------------------

The following variables are used throughout:

* `n` — number of data samples
* `m` — number of neurons (grows dynamically; refers to the final neuron count)
* `d` — data dimension
* `e` — number of training epochs

Space complexity
****************

The weight matrix and the input data each contribute :math:`O(m \cdot d)` and :math:`O(n \cdot d)`.
Additionally, the Floyd–Warshall distance matrix between all neuron pairs is stored as an
:math:`m \times m` array, adding :math:`O(m^2)`:

.. math::
    S = O(d \cdot (m + n) + m^2)

The :math:`m^2` term dominates the space cost when `d` is small relative to `m`.

Time complexity (fit)
*********************

Each training epoch involves three operations with distinct costs:

**BMU search** — for every sample, the nearest neuron is found by comparing against all `m`
weight vectors across `d` dimensions (implemented as a fused Numba JIT kernel):

.. math::
    T_{\text{BMU}} = O(n \cdot m \cdot d) \text{ per epoch}

**Batch weight update** — the new weights are computed via a matrix multiplication of the
:math:`m \times m` neighbourhood kernel with the :math:`m \times d` Voronoi centre matrix:

.. math::
    T_{\text{update}} = O(m^2 \cdot d) \text{ per epoch}

**Floyd–Warshall recomputation** — whenever a new neuron is inserted the graph distance
matrix is recomputed from scratch. With up to `m` growth steps and an :math:`O(m^3)` cost
per step, the cumulative growth overhead is:

.. math::
    T_{\text{growth}} = O(m^4)

The total fit complexity is therefore:

.. math::
    T_{\text{fit}} = O\!\left(e \cdot (n \cdot m \cdot d + m^2 \cdot d) + m^4\right)

* When :math:`n \gg m` the BMU search dominates: :math:`O(n \cdot m \cdot d \cdot e)`.
* When :math:`m \gg n` the weight update dominates: :math:`O(m^2 \cdot d \cdot e)`.
* The :math:`m^4` Floyd–Warshall term is independent of `n` and `e` and is bounded by
  ``max_neurons``.

Time complexity (predict)
*************************

Prediction requires only a single BMU search over the fitted weight matrix:

.. math::
    T_{\text{predict}} = O(n \cdot m \cdot d)

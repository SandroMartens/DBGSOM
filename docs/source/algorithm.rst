Algorithm
=========




Overview
--------

The SOM algorithm performs in two steps. First, competition among the neurons to find the winner and second, adaptation of the weight vector of the winner neuron and its topological neighbors. Instead of being confined to a predetermined number of neurons, DBGSOM offers a flexible structure and requires fewer epochs compared to the original SOM, which enables the ability to learn the nonlinear manifolds in high-dimensional feature space.




Batch Learning Algorithm
------------------------

Each training epoch consists of two steps: BMU search and weight update.

BMU Search
^^^^^^^^^^

Let the input data items be :math:`n`-dimensional vectors :math:`x`. Let the codebook
vectors (neuron weights) be :math:`m_i`, indexed by :math:`i`. The winner neuron
:math:`c` --- the *best matching unit* (BMU) --- is the one with the smallest distance
from :math:`x`:

.. math::
    c = \operatorname{argmin}_i \{ \lVert x - m_i \rVert \}

All input vectors are presented simultaneously; each is assigned to its nearest neuron.

Weight Update
^^^^^^^^^^^^^

The new weights are calculated as the neighborhood-weighted mean of all samples
assigned to each neuron:

.. math::
    w_i^{new} = \frac{\sum_{j=1}^{k}h_{c_{j, i}}  \cdot x_j}{\sum_{j=1}^{k}h_{c_{j, i}}}

where :math:`h_{c_{j, i}}` is the neighborhood function (see below).
The neighborhood bandwidth starts large (default: 20% of the map's larger side) and
decays toward a small end value (default: 5%) over the course of training.

This algorithm repeats for a given number of iterations or until the weight vectors
no longer change between iterations.
Neighborhood Functions
^^^^^^^^^^^^^^^^^^^^^^

Two neighborhood functions are available via the ``neighborhood_function`` parameter:

**Gaussian** (default, ``"gaussian"``)

.. math::
    h_{c_{j, i}} = \exp \left(- \frac{d_{ij}^2}{2{\sigma}^2}\right)

where :math:`d_{ij}` is the graph distance between neurons :math:`i` and :math:`c_j` on the SOM grid.

**Cut Gaussian** (``"cutgauss"``)

Same as Gaussian, but set to zero for all neuron pairs with graph distance :math:`d_{ij} > 2\sigma`. This concentrates updates on a well-defined neighborhood and suppresses long-range interference.




Decay Functions
^^^^^^^^^^^^^^^
The decay from ``sigma_start`` to ``sigma_end`` follows either an **exponential** or **linear** schedule (``decay_function`` parameter). For exponential decay the learning rate is chosen so that 99% of the drop from ``sigma_start`` to ``sigma_end`` is completed by the end of the coarse phase.




Distance Metrics
^^^^^^^^^^^^^^^^

Two distance metrics are supported via the ``metric`` parameter:

- ``"euclidean"`` (default): standard Euclidean distance; BMU search via BLAS ``euclidean_distances``.
- ``"cosine"``: cosine dissimilarity :math:`1 - \langle x, w \rangle`; data and weights are L2-normalised before training and at each update step.




DBGSOM Algorithm
----------------
Training of the DBGSOM starts from a small number of initial neurons to a larger map by adding new neurons to the network. A batch growing approach for SOM called Directed Batch Growing Self-Organizing Map is used. It uses the accumulative error of the neurons on the grid to direct the growing phase in terms of position and weight initialization of new neurons. After each learning iteration new neurons can be added from boundaries by filling one of the adjacent free positions and assigning a proper weight vector. This implements :ref:`Assumption 2 <assumption-2>` and :ref:`Assumption 3 <assumption-3>`.




Directed Horizontal Growth
^^^^^^^^^^^^^^^^^^^^^^^^^^

Growing Threshold
"""""""""""""""""

The growing threshold ``GT`` is computed via the statistics-enhanced formula (Qu et al. 2019, Eq. 5):

.. math::
    GT = \lambda \cdot \left\lVert \operatorname{std}(X) \right\rVert

where :math:`\operatorname{std}(X)` is the vector of per-feature standard deviations and
:math:`\lambda` is set via ``lambda_`` (default 115.0, paper-optimized). This makes GT directly
comparable in scale to the per-neuron cumulative error.

Reference: Qu et al., *Statistics-enhanced Direct Batch Growth Self-Organizing Mapping for efficient DoS Attack Detection*, IEEE Access, 2019.

Estimating the equilibrium neuron count
""""""""""""""""""""""""""""""""""""""""

The GT formula implies a statistical estimate of the final map size before training.
A neuron *i* triggers growth when its accumulated squared error exceeds GT:

.. math::
    \sum_{j:\,c_j = i} \lVert x_j - w_i \rVert^2 > GT

At convergence, the growth trigger for a boundary neuron :math:`b` is:

.. math::
    E_b + \frac{1}{2} \sum_{i \in \mathcal{N}(b),\, i \text{ interior}} E_i > GT

where :math:`\mathcal{N}(b)` are the neighbours of :math:`b` (see `Growth Triggering`_).
Assuming approximately uniform hit distribution (:math:`n_i \approx n / K`) so that
:math:`E_b \approx E_i \approx E`, the boundary condition at the tipping point becomes:

.. math::
    E + \frac{E}{2} = \frac{3E}{2} \approx GT \implies E \approx \frac{2}{3}\,GT

Every neuron therefore sits just below :math:`\tfrac{2}{3}\,GT` at convergence.
Substituting :math:`E \approx \tfrac{n}{K} \cdot \overline{QE} = \tfrac{2}{3}\,GT`:

.. math::
    K_{\mathrm{eq}} \approx \frac{3}{2} \cdot \frac{n \cdot \overline{QE}}{\lambda \cdot \lVert \operatorname{std}(X) \rVert}

where :math:`\overline{QE}` is the mean quantization error of the current map.
Because :math:`\overline{QE}` itself decreases as the map grows, the formula is
circular; substituting the early-stage QE (e.g. with the initial 4 neurons)
gives a conservative upper bound on the natural stopping point.

**Qualitative effects**

+--------------------+----------+--------------------------------------+
| Change             | Effect   | Reason                               |
+====================+==========+======================================+
| ``lambda_`` ↑      | K ↓      | GT rises — harder to exceed          |
+--------------------+----------+--------------------------------------+
| training set N ↑   | K ↑      | more samples accumulate error faster |
+--------------------+----------+--------------------------------------+
| ``‖std(X)‖`` ↑     | K ↓      | GT rises proportionally              |
+--------------------+----------+--------------------------------------+

For data on an isotropic Gaussian manifold with intrinsic dimension :math:`d`,
the within-cell variance scales as :math:`\overline{QE} \sim \lVert\operatorname{std}(X)\rVert \cdot K^{-1/d}`.
Substituting into the equilibrium equation yields the closed-form growth law
(the :math:`\tfrac{3}{2}` prefactor does not affect the exponent):

.. math::
    K_{\mathrm{eq}} \sim \left(\frac{n}{\lambda}\right)^{d/(d+1)}

Growth is therefore **sub-linear in** :math:`n`, consistent with the empirical
observation that doubling the dataset does not double the map size.

.. note::
   The :math:`\tfrac{3}{2}` factor derives from the single-interior-neighbour
   case and is exact when each boundary neuron has exactly one interior
   neighbour.  Corner neurons (zero interior neighbours) experience no
   redistribution and sit closer to :math:`GT`; deep interior neurons with
   multiple boundary neighbours distribute their error across several
   recipients, slightly reducing the effective per-neuron contribution.
   For typical rectangular maps these effects cancel to first order, so the
   :math:`\tfrac{3}{2}` coefficient is a good approximation.  The asymptotic
   scaling :math:`K \sim (n/\lambda)^{d/(d+1)}` remains valid as
   :math:`K \to \infty` since the boundary fraction :math:`O(1/\sqrt{K})`
   vanishes.

**Runtime consequence.** Substituting :math:`K_{\mathrm{eq}}` for :math:`m` in the
fit complexity :math:`O\!\left(e \cdot n \cdot m \cdot d + m^4\right)` gives an
approximate total cost of:

.. math::
    T_{\mathrm{fit}} \approx O\!\left(e \cdot n^{(2d+1)/(d+1)} \cdot d \cdot \lambda^{-d/(d+1)}\right)

Increasing ``lambda_`` therefore reduces both the neuron count **and** the
dominant training cost roughly as :math:`\lambda^{-d/(d+1)}`.

Growth Triggering
"""""""""""""""""

Unlike the original DBGSOM paper — where new neurons are added after every
epoch — this implementation uses **convergence-triggered growth**: the map
trains within a single convergence cycle until the convergence criterion is
met, then executes a growth step and begins the next cycle with a decayed
:math:`\sigma`. After each batch weight update, the change in the weight matrix is compared against ``convergence_threshold``. When the change falls below this threshold, the map is considered converged and — if still in the coarse phase and below ``max_neurons`` — a growth step is performed:

1. The accumulative error :math:`E_i` for each neuron is evaluated.
2. For each non-boundary neuron :math:`n_i` where :math:`E_i > GT`, half its error is distributed to neighboring boundary neurons.
3. New neurons are inserted at all boundary positions where :math:`E_i > GT`, starting with the highest-error neuron.

Waiting for convergence before inserting neurons is deliberate: only once the map has converged do the per-neuron error values constitute stable estimates of the true quantization error. Growing from a transient training state would make the error distribution unreliable and could direct new neurons to the wrong positions (Kohonen, 2014).

The position and weight of each new neuron are determined by the directed insertion rule (Vasighi and Amini, Section 3.3.1.1): the free adjacent position whose corner neighbor has the highest error is selected, and the new weight is initialized by reflecting the opposite neighbor through the boundary neuron. Because each inserted neuron inherits a weight derived from its already-trained neighbours, the map remains partially ordered after growth. A partially ordered map is empirically known to converge significantly faster than a randomly initialised one (Kohonen, 2001), which is why the weight initialisation rule is central to the algorithm's efficiency.

After a growth step, :math:`\sigma` is updated via the decay function and ``converged_`` is reset to ``False``, starting the next convergence cycle.




Implementation Details
----------------------

The following assumptions from the literature motivate the implementation choices described below.




Assumptions
^^^^^^^^^^^
.. _assumption-1:

**Assumption 1** *(Finite convergence)* —
A network with constant neighborhood bandwidth :math:`\sigma` always converges in a finite number of iterations. Empirical Evidence.

.. _assumption-2:

**Assumption 2** *(Small-network speed)* —
A small network converges faster than a large one. Follows directly from :ref:`complexity`

.. _assumption-3:

**Assumption 3** *(Topology aids convergence)* —
A partially ordered map converges significantly faster than a randomly initialised one. Empirical Evidence.




Convergence Cycles
^^^^^^^^^^^^^^^^^^

Training is structured as a sequence of **convergence cycles**. Each cycle trains at fixed
:math:`\sigma` until the convergence criterion is met. In the *coarse phase* a growth step
fires at the end of the cycle and :math:`\sigma` decays before the next cycle begins. In the
*fine phase* no growth occurs and :math:`\sigma` is fixed.

This is inspired by the classical two-phase SOM training (Kohonen, 2001), but rather than a
hard split into two monolithic phases, each phase is realised as one or more convergence
cycles. The coarse phase spans the first ``coarse_training_frac * n_iter`` epochs; the
remaining epochs form the fine phase.

- **Coarse phase** (growth cycles): :math:`\sigma` starts at ``sigma_start``
  (default :math:`0.2 \cdot \sqrt{m}`) and decays toward ``sigma_end``
  (default :math:`0.05 \cdot \sqrt{m}`) after each growth step.
- **Fine phase** (refinement cycle): no further growth. :math:`\sigma` is fixed
  to ``sigma_fine`` if set, otherwise ``sigma_end`` is used.

This is consistent with :ref:`Assumption 1 <assumption-1>`: because :math:`\sigma` is
constant within each cycle, finite convergence is guaranteed for every cycle including the
final fine phase.




Sigma Schedule
^^^^^^^^^^^^^^
Between growth steps :math:`\sigma` is held **constant** within each convergence
cycle. Only when the map converges and a growth step fires does :math:`\sigma`
advance to its next decayed value. This is consistent with :ref:`Assumption 1 <assumption-1>` (constant
:math:`\sigma` guarantees finite convergence within each cycle), while the
neighbourhood shrinks progressively as the map grows.

Per :ref:`Assumption 3 <assumption-3>`, the map needs to be ordered before
growth — topological ordering is a strictly weaker condition than weight
convergence. This follows from the fact that zero winner changes between
epochs implies an identical weight update, hence no weight change (the map
has converged). Ordering is therefore achieved before convergence, and
growing from an ordered-but-not-converged map is sufficient.

Computing the topographic error (TE) each epoch is not used as the ordering
criterion, even though in full-search mode it is nearly free (the second BMU
is a by-product of the distance scan). The reason is that TE does not signal
readiness to grow: it measures topological quality, not weight stability, and
it is non-monotone — TE rises transiently after each growth step as new
neurons are inserted. Winner-stability is a better proxy for the convergence
cycle endpoint: when fewer than ``winner_stability_threshold`` of samples
change their BMU between epochs, the Voronoi regions are stable, which in
practice coincides with low TE and indicates the map is ready for the next
growth step.




Robustness Weighting
^^^^^^^^^^^^^^^^^^^^

Before the batch weight update, each sample :math:`x_j` is assigned a robustness weight

.. math::
    s_j = 1 - \left(1 - \exp\!\left(-\gamma \, \lVert x_j - w_{c_j} \rVert^2\right)\right)^{1/2}, \quad \gamma = \frac{1}{\operatorname{Var}(X)}

Samples far from their BMU (outliers) receive lower weights, making the map more robust to noise. Samples close to their BMU receive a weight near 1.

Reference: D'Urso et al., *Smoothed self-organizing map for robust clustering*, Information Sciences, 2019.




Accelerated BMU Search
^^^^^^^^^^^^^^^^^^^^^^

BMU search dominates runtime for large maps (see :ref:`complexity`). To reduce
this cost, a pointer-based search restricts each sample's winner search to the
neuron that won in the previous epoch and its graph neighbours.

Controlled by ``pointer_search``:

- ``"none"``: full search over all neurons — :math:`O(n \cdot m \cdot d)` per epoch.
- ``"fine"`` (default): pointer search only during the fine training phase, where
  the map is stable and winners rarely move beyond the local neighbourhood.
  Near-identical quality at approximately 3× speedup.
- ``"all"``: pointer search in both phases. Improves topographic error (map
  topology is more locally consistent) but reduces quantization accuracy.
  Use a larger ``pointer_search_radius`` to recover quality.

The search radius is set via ``pointer_search_radius`` (default 1). Candidates
are all neurons within that many graph hops of the previous winner, read from
the already-computed distance matrix.

For a 2D grid SOM the number of candidate neurons at radius :math:`r` is
approximately :math:`2r^2 + 2r + 1`, giving a speedup of roughly
:math:`m \,/\, (2r^2 + 2r + 1)` relative to the full search.




Winner-Stability Convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``winner_stability_threshold`` is set (default 0.01), the coarse phase
uses the winner-change rate as its convergence criterion instead of weight-delta:

.. math::

    \text{converged} \iff
    \frac{\left|\left\{ j : c_j^{(t)} \neq c_j^{(t-1)} \right\}\right|}{n} < \tau_w

where :math:`\tau_w` = ``winner_stability_threshold`` and :math:`c_j^{(t)}` is
the BMU of sample :math:`j` at epoch :math:`t`. This criterion responds faster
to map stabilisation than weight-delta and is well-matched to pointer search
(stable winners are exactly the signal pointer search relies on).
Set ``winner_stability_threshold=None`` to revert to weight-delta convergence.
This criterion determines when a convergence cycle ends and a growth step may fire.




First Classification
--------------------

For sample classification, each neuron :math:`n_i` gets assigned a label :math:`L_i` as the most common class label :math:`l` of all samples represented by that prototype:

.. math::
    L_i = \operatorname{mode}(l_1, \ldots, l_n)




Extensions
----------

There is currently one extension to the original DBGSOM implemented:

- Hierarchical SOM (HSOM)

The HSOM handles densely clustered data samples that cannot be distinguished by further neuron growth. A new, smaller SOM is created for neurons whose error remains high after the horizontal growth phase.

Hierarchical DBGSOM
^^^^^^^^^^^^^^^^^^^

After the horizontal growth phase, each neuron :math:`n_i` whose quantization error exceeds a vertical growing threshold triggers the creation of a child SOM trained on all samples mapped to :math:`n_i`. The vertical growing threshold is:

.. math::
    VGT = \tau_2 \cdot QE_0

where :math:`\tau_2` is the ``tau_2`` parameter (default 0.5) and :math:`QE_0` is the quantization error of a single-neuron SOM whose weight equals the mean of all training data. This formulation follows the GHSOM stopping criterion.

A child SOM is only created when the number of samples mapped to the neuron exceeds ``min_samples_vertical_growth``.

Reference: Qu et al., *Entropy-Defined Direct Batch Growing Hierarchical Self-Organizing Mapping for Efficient Network Anomaly Detection*, IEEE Access, 2021.




.. _complexity:

Algorithmic Complexity
----------------------
Algorithmic complexity
----------------------
The following variables are used throughout:

* `n` — number of data samples
* `m` — number of neurons (grows dynamically; refers to the final neuron count)
* `d` — data dimension
* `e` — number of training epochs
* `r` - Neighborhood width when using pointer search

Training
^^^^^^^^

Space complexity
""""""""""""""""

The weight matrix and the input data each contribute :math:`O(m \cdot d)` and :math:`O(n \cdot d)`.
Additionally, the Floyd–Warshall distance matrix between all neuron pairs is stored as an
:math:`m \times m` array, adding :math:`O(m^2)`.
The **stored** state after training is therefore:

.. math::
    S_{\text{stored}} = O(d \cdot (m + n) + m^2)

During training, the full-search BMU path (``pointer_search="none"``, coarse phase, or
``similarity="cosine"``) allocates an :math:`n \times m` pairwise distance matrix,
adding a peak term of :math:`O(n \cdot m)`:

.. math::
    S_{\text{peak}} = O(d \cdot (m + n) + m^2 + n \cdot m)

For large datasets this :math:`n \times m` allocation dominates.
The pointer-based fine-phase search (``pointer_search="fine"``, default) avoids this
allocation entirely by searching only the previous winner and its graph neighbours.

Time complexity
"""""""""""""""

Each training epoch involves three operations with distinct costs:

**BMU search** — for every sample, the nearest neuron is found by comparing against all `m`
weight vectors across `d` dimensions (Euclidean: BLAS ``euclidean_distances``; cosine: fused
Numba JIT dot-product kernel):

.. math::
    T_{\text{BMU}} = O(n \cdot m \cdot d) \text{ per epoch}

With ``pointer_search="fine"`` (default), the fine-phase search is restricted to
:math:`O(r^2)` candidates per sample, reducing the fine-phase BMU cost to
:math:`O(n \cdot r^2 \cdot d \cdot e_{\text{fine}})`, independent of `m`.

**Batch weight update** — two steps per epoch:

1. **Voronoi centre computation** (Numba JIT) — for each neuron, compute the kernel-weighted
   mean of all :math:`n` samples assigned to it: :math:`O(n \cdot d)`.
2. **Neighbourhood matrix multiply** (BLAS) — contract the :math:`m \times m` neighbourhood
   kernel with the :math:`m \times d` Voronoi centre matrix: :math:`O(m^2 \cdot d)`.

.. math::
    T_{\text{update}} = O(n \cdot d + m^2 \cdot d) \text{ per epoch}

When :math:`n \gg m^2 / d` the Voronoi step dominates; otherwise the matrix multiply does.

**Floyd–Warshall recomputation** — whenever a new neuron is inserted the graph distance
matrix is recomputed from scratch. With up to `m` growth steps and an :math:`O(m^3)` cost
per step, the cumulative growth overhead is:

.. math::
    T_{\text{growth}} = O(m^4)

The total fit complexity is therefore:

.. math::
    T_{\text{fit}} = O\!\left(e \cdot (n \cdot m \cdot d + n \cdot d + m^2 \cdot d) + m^4\right)

Simplified (absorbing :math:`n \cdot d` into :math:`n \cdot m \cdot d` since :math:`m \geq 1`):

.. math::
    T_{\text{fit}} = O\!\left(e \cdot (n \cdot m \cdot d + m^2 \cdot d) + m^4\right)

* When :math:`n \gg m` the BMU search dominates: :math:`O(n \cdot m \cdot d \cdot e)`.
* When :math:`m \gg n` the weight update dominates: :math:`O(m^2 \cdot d \cdot e)`.
* The :math:`m^4` Floyd–Warshall term is independent of `n` and `e` and is bounded by
  ``max_neurons``.

Runtime
^^^^^^^

Prediction requires only a single BMU search over the fitted weight matrix:

.. math::
    T_{\text{predict}} = O(n \cdot m \cdot d)

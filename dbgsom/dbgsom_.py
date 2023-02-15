"""
DBGSOM: Directed Batch Growing Self Organizing Map
"""

import sys
from math import log
from statistics import mode
from typing import Any

try:
    import networkx as nx
    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    import seaborn.objects as so
    from sklearn.base import (
        BaseEstimator,
        ClassifierMixin,
        ClusterMixin,
        TransformerMixin,
    )
    from sklearn.metrics import pairwise_distances
    from sklearn.utils import check_array, check_random_state, check_X_y
    from sklearn.utils.validation import check_is_fitted
    from tqdm import tqdm
except ImportError as e:
    print(e)
    sys.exit()


# pylint:  disable= attribute-defined-outside-init
class DBGSOM(BaseEstimator, ClusterMixin, TransformerMixin, ClassifierMixin):
    """A Directed Batch Growing Self-Organizing Map.

    Parameters
    ----------
    sf : float, default = 0.1
        Spreading factor to calculate the treshold for neuron insertion

        0 means no growth, 1 means unlimited growth

        0 < sf < 1.

    n_epochs_max : int, default = 50
        Maximal Number of training epochs.

    max_neurons : int, default = 100
        Maximum number of neurons in the som.

    decay_function : {'exponential', 'linear'}, default = 'exponential'
        Decay function to use for neighborhood bandwith sigma.

    coarse_training_frac : float, default = 0.5
        Fraction of training epochs to use for coarse training.
        Training happens in two phases, coarse and fine training. In coarse training,
        the neighborhood bandwidth is decreased from sigma_start to sigma_end and
        the network grows according to the growing rules. In fine training, the
        bandwidth is constant at sigma_end and no new neurons are added.

    error_method : {"distance", "entropy"}
        Method for calculating the error of neurons and samples.

        "distance" : The cumulative error is the sum of individual
        error.

        "entropy": For supervised learning we can use the entropy
        of labels of the samples as error.

    metric : str, default = euclidean
        The metric to use for computing distances between prototypes and samples.

    random_state : any (optional), default = None
        Random state for weight initialization.

    convergence_treshold : float, default = 10 ** -10
        If the sum of all weight changes is smaller than the threshold,
        convergence is assumed and the training is stopped.

    threshold_method : {"classical", "se"}
        Method to calculate the growing threshold.

        "classical" : Threshold is only dependent on the dimension of the input data.

        `gt = n_dim * -log(sf)`

        "se" : Statistics enhanced formula, which uses the standard
        deviation of features in X.

        `gt = lambda * np.sqrt(np.sum(np.std(X, axis=0, ddof=1) ** 2))`

    sigma_start : {None, numeric}, default = None
        Start for the neighborhood bandwidth.

        If `None`, it is calculated dynamically in each epoch as

        `sigma_start = 0.2 * sqrt(n_neurons)`.

    sigma_end : {None, numeric}, default = None
        End for the neighborhood bandwidth.

        If `None` , it is calculated dynamically in each epoch as

        `sigma_end = max(0.7, 0.05 * sqrt(n_neurons))`.

    Attributes
    ----------
    som_ : NetworkX.graph
        Graph object containing the neurons with attributes

    weights_ : ndarray of shape (n_prototypes, n_features)
        Learned weights of the neurons

    topographic_error_ : float
        Fraction of training samples where the first and second best matching
        prototype are not neighbors on the SOM

    quantization_error_ : float
        Average distance from all training samples to their nearest prototypes


    """

    def __init__(
        self,
        n_epochs_max: int = 50,
        sf: float = 0.1,
        sigma_start: float | None = None,
        sigma_end: float | None = None,
        decay_function: str = "exponential",
        coarse_training_frac: float = 0.5,
        random_state: Any = None,
        convergence_treshold: float = 10**-10,
        max_neurons: int = 100,
        metric: str = "euclidean",
        threshold_method: str = "classical",
        error_method: str = "distance",
    ) -> None:
        self.sf = sf
        self.n_epochs_max = n_epochs_max
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.decay_function = decay_function
        self.coarse_training_frac = coarse_training_frac
        self.random_state = random_state
        self.convergence_treshold = convergence_treshold
        self.max_neurons = max_neurons
        self.metric = metric
        self.threshold_method = threshold_method
        self.error_method = error_method

    def fit(self, X: npt.ArrayLike, y: None | npt.ArrayLike = None):
        """Train SOM on training data.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : dbgsom
            Trained estimator
        """
        if y is None:
            X = check_array(array=X, ensure_min_samples=4)
            self._y_is_fitted = False
        else:
            X, y = check_X_y(X=X, y=y, ensure_min_samples=4)
            self._y_is_fitted = True
            self.classes_, y = np.unique(y, return_inverse=True)
        self.random_state_ = check_random_state(self.random_state)
        self._initialization(X)
        self._grow(X, y)
        # self.rep = self._calculate_rep(X)
        if self._y_is_fitted:
            self._label_prototypes(X, y)
        self.topographic_error_ = self._topographic_error_func(X)
        self.quantization_error_ = self.calculate_quantization_error(X)
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X) -> np.ndarray:
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            If fitted unsupervised: Index of best matching prototype.

            If fitted supervised: Label of the predicted class.
        """
        check_is_fitted(self)
        X = check_array(array=X, dtype=[float, int])
        if not self._y_is_fitted:
            labels = self._get_winning_neurons(X, n_bmu=1)
        else:
            bmus = self._get_winning_neurons(X, n_bmu=1)
            labels = []
            for bmu in bmus:
                label = self.som_.nodes[self.neurons_[bmu]]["label"]
                if label is not None:
                    labels.append(label)
                else:
                    labels.append(-1)

            labels = np.array(labels)

        return self.classes_[labels]

    def transform(self, X: npt.ArrayLike, y=None) -> np.ndarray:
        """Calculate the distance matrix of all samples and prototypes.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data to transform

        Returns
        -------
        distances : np.ndarray
            Distance matrix of shape (n_protypes, n_samples)
        """
        check_is_fitted(self)
        X = check_array(X)
        distances = pairwise_distances(
            self.weights_, X, metric=self.metric, n_jobs=-1
        ).T
        return distances

    def plot(self, attribute: None | str = None, palette="magma_r") -> None:
        """Plot the neurons.

        Parameters
        ----------
        attribute : {None, "epoch_created", "error", "distances"}, default = None
            Attribute which is represented as color.

            "epoch_created" : When the neuron was created.

            "error" : Quantization error of each neuron.

            "distances" : Average distance to neighbor neurons in
            the input space. Creates a U-Matrix.

        palette : matplotlib colormap/seaborn palette, default = "magma_r"
            Name of seaborn palette to color code the values of attribute
        """

        dots = pd.DataFrame(np.array(self.neurons_), columns=["x", "y"])
        dots["epoch_created"] = list(
            dict(self.som_.nodes.data("epoch_created")).values()
        )
        dots["error"] = list(dict(self.som_.nodes.data("error")).values())
        dots["distances"] = self._get_u_matrix()
        dots["label"] = list(dict(self.som_.nodes.data("label")).values())
        so.Plot(dots, x="x", y="y", color=attribute).add(so.Dot()).scale(
            color=palette
        ).show()

    def _get_u_matrix(self) -> list:
        """Calculate the average distance from each neuron to it's neighbors."""

        g = self.som_
        distances = []
        for node, neighbors in g.adj.items():
            node_weight = g.nodes[node]["weight"]
            distance = 0
            for neighbor in neighbors:
                nbr_weight = g.nodes[neighbor]["weight"]
                distance += np.linalg.norm(node_weight - nbr_weight)
            distances.append(distance / len(neighbors))

        return distances

    def _calculate_rep(self, X: npt.ArrayLike):
        """Return the resemble entropy parameter.

        1. Calculate histogram of components of each sample.
        2. Calculate entropy of each sample from histogram
        3. Save minimum and maximum rep for all classes

        Use 20 bins as default"""

        hists = []
        for sample in X:
            hists.append(np.histogram(sample, bins=20)[0])

    def _initialization(self, data: npt.NDArray) -> None:
        """First training phase.

        Calculate growing threshold according to the argument. Create
        a graph containing the first four neurons in a square with
        init vectors.
        """
        self._current_epoch = 0
        self.converged_ = False
        self._training_phase = "coarse"
        data = data.astype(np.float32)
        self.growing_threshold_ = self._calculate_growing_threshold(data)

        self.som_ = self._create_som(data)
        self._distance_matrix = nx.floyd_warshall_numpy(self.som_)
        self.weights_ = np.array(list(dict(self.som_.nodes.data("weight")).values()))
        self.neurons_ = list(self.som_.nodes)

    def _calculate_growing_threshold(self, data):
        if self.threshold_method == "classical":
            n_dim = data.shape[1]
            gt = -n_dim * log(self.sf)

        elif self.threshold_method == "se":
            gt = log(self.sf) * np.sqrt(np.sum(np.std(data, axis=0, ddof=1) ** 2))

        return gt

    def _grow(self, data: npt.NDArray, y) -> None:
        """Second training phase"""
        for current_epoch in tqdm(
            iterable=range(self.n_epochs_max),
            unit=" epochs",
        ):
            self._current_epoch = current_epoch
            if current_epoch > self.coarse_training_frac * self.n_epochs_max:
                self._training_phase = "fine"
            self.weights_ = np.array(
                list(dict(self.som_.nodes.data("weight")).values())
            )
            # check if new neurons were inserted
            if len(self.som_.nodes) > len(self.neurons_) or current_epoch == 0:
                self.neurons_ = list(self.som_.nodes)
                self._update_distance_matrix()

            winners = self._get_winning_neurons(data, n_bmu=1)
            self._update_weights(winners, data)
            if self.converged_:
                break

            self._write_accumulative_error(winners, data, y)
            if (
                current_epoch != self.n_epochs_max
                and self._training_phase == "coarse"
                and len(self.neurons_) < self.max_neurons
                and current_epoch % 5 == 3
            ):
                self._distribute_errors()
                self._add_new_neurons()

    def _create_som(self, data: npt.NDArray) -> nx.Graph:
        """Create a graph containing the first four neurons in a square.
        Each neuron has a weight vector randomly chosen from the training
         samples."""
        rng = np.random.default_rng(seed=self.random_state)
        init_vectors = rng.choice(a=data, size=4, replace=False)
        neurons = [
            ((0, 0), {"weight": init_vectors[0], "epoch_created": 0}),
            ((0, 1), {"weight": init_vectors[1], "epoch_created": 0}),
            ((1, 0), {"weight": init_vectors[2], "epoch_created": 0}),
            ((1, 1), {"weight": init_vectors[3], "epoch_created": 0}),
        ]

        #  Build a square
        edges = [
            ((0, 0), (0, 1)),
            ((0, 0), (1, 0)),
            ((1, 0), (1, 1)),
            ((0, 1), (1, 1)),
        ]

        som = nx.Graph()
        som.add_nodes_from(neurons)
        som.add_edges_from(edges)

        return som

    def _get_winning_neurons(self, data: npt.NDArray, n_bmu: int) -> np.ndarray:
        """Calculate distances from each neuron to each sample.

        Return index of winning neuron or best matching units(s) for each
        sample.
        """
        weights = self.weights_
        distances = pairwise_distances(X=weights, Y=data, metric=self.metric, n_jobs=-1)
        if n_bmu == 1:
            winners = np.argmin(distances, axis=0)
        else:
            winners = np.argsort(distances, axis=0)[:n_bmu]

        return winners

    def _label_prototypes(self, X, y):
        winners = self._get_winning_neurons(X, n_bmu=1)
        for winner_index, neuron in enumerate(self.neurons_):
            labels = y[winners == winner_index]
            # dead neuron
            if len(labels) == 0:
                label_winner = -1
            else:
                label_winner = mode(labels)
            self.som_.nodes[neuron]["label"] = int(label_winner)

    def _update_distance_matrix(self) -> None:
        """Update distance matrix between neurons.
        Only paths of length =< 3 * sigma + 1 are considered for performance
        reasons.
        """
        n_neurons = len(self.neurons_)
        distance_matrix = np.zeros((n_neurons, n_neurons))
        distance_matrix.fill(np.inf)
        sigma = self._sigma()
        dist_dict = dict(
            nx.all_pairs_shortest_path_length(self.som_, cutoff=3 * sigma + 1)
        )
        for i1, neuron1 in enumerate(self.neurons_):
            for i2, neuron2 in enumerate(self.neurons_):
                if neuron2 in dist_dict[neuron1].keys():
                    distance_matrix[i1, i2] = dist_dict[neuron1][neuron2]

        self._distance_matrix = distance_matrix

    # @profile
    def _update_weights(self, winners: np.ndarray, data: npt.NDArray) -> None:
        """Update the weight vectors according to the batch learning rule.

        Step 1: Calculate the center of the voronoi set of each neuron.
        Step 2: Count the number of samples in each voronoi set.
        Step 3: Calculate the kernel function for all neuron pairs.
        Step 4: calculate the new weight vectors as
            New weight vector = sum(kernel * n_samples * centers)
                / sum(kernel * n_samples)
        Step 5: Write new weight vectors to the graph.
        """
        # new
        # Sadly we cant use the easy indexing with numpy because thats too slow
        # see https://stackoverflow.com/questions/75423927/what-is-the-fastest-way-to-select-multiple-elements-from-a-numpy-array/75424204#75424204
        voronoi_set_centers_sum = np.zeros_like(self.weights_)
        center_counts = np.zeros(shape=(len(self.weights_),))

        for sample, winner in zip(data, winners):
            voronoi_set_centers_sum[winner] += sample
            center_counts[winner] += 1
        # No div by 0
        center_counts = np.maximum(center_counts, 1)
        voronoi_set_centers = voronoi_set_centers_sum / center_counts[:, None]

        # old
        # step 1
        # voronoi_set_centers = np.zeros_like(self.weights_)
        # for winner in np.unique(winners):
        #     voronoi_set_centers[winner] = data[winners == winner].mean(axis=0)

        # step 2
        neuron_activations = np.zeros(shape=len(self.neurons_), dtype=np.float32)
        winners, winner_counts = np.unique(winners, return_counts=True)
        for winner, count in zip(winners, winner_counts):
            neuron_activations[winner] = count

        # Step 3
        gaussian_kernel = self._gaussian_neighborhood()

        # Step 4
        new_weights = np.sum(
            voronoi_set_centers
            * neuron_activations[:, np.newaxis]
            * gaussian_kernel[:, :, np.newaxis],
            axis=1,
        ) / np.sum(
            gaussian_kernel[:, :, np.newaxis] * neuron_activations[:, np.newaxis],
            axis=1,
        )

        # Step 5
        new_weights_dict = dict(zip(self.neurons_, new_weights))
        change = np.linalg.norm(self.weights_ - new_weights, axis=1)
        change_total = np.sum(change)
        if change_total < self.convergence_treshold:
            self.converged_ = True
        nx.set_node_attributes(G=self.som_, values=new_weights_dict, name="weight")

    def _gaussian_neighborhood(self) -> np.ndarray:
        """Calculate the gaussian neighborhood function for all neuron
        pairs using the distance matrix."""
        sigma = self._sigma()
        h = np.exp(-(self._distance_matrix**2 / (2 * sigma**2))).astype(np.float32)

        return h

    # @profile
    def _write_accumulative_error(
        self, winners: np.ndarray, data: npt.NDArray, y
    ) -> None:
        """Get the quantization error for each neuron
        and save it as "error" attribute of each node.
        """
        if self.error_method == "entropy":
            for winner_index, neuron in enumerate(self.neurons_):
                _, counts = np.unique(y[winners == winner_index], return_counts=True)
                total = np.sum(counts)
                counts = counts / total
                error = np.sum(-counts * np.log(counts))
                self.som_.nodes[neuron]["error"] = error

        else:
            errors = np.zeros(shape=len(self.weights_))
            for sample, winner in zip(data, winners):
                error = np.linalg.norm(self.weights_[winner] - sample)
                errors[winner] += error
            for i, error in enumerate(errors):
                neuron = self.neurons_[i]
                self.som_.nodes[neuron]["error"] = error

    def _distribute_errors(self) -> None:
        """For each neuron i which is not a boundary neuron and E_i > GT,
        a half value of E_i is equally distributed to the neighboring
        boundary neurons, if exist.
        """
        for node, neighbors in self.som_.adj.items():
            if len(neighbors.items()) == 4:
                is_boundary = False
            else:
                is_boundary = True
            node_error = self.som_.nodes[node]["error"]

            if not is_boundary and node_error > self.growing_threshold_:
                n_boundary_neighbors = 0
                for neighbor in neighbors.keys():
                    if len(self.som_.adj[neighbor].items()) < 4:
                        n_boundary_neighbors += 1

                for neighbor in neighbors.keys():
                    if len(self.som_.adj[neighbor].items()) < 4:
                        self.som_.nodes[neighbor]["error"] += (
                            0.5 * node_error / n_boundary_neighbors
                        )
                self.som_.nodes[node]["error"] /= 2

    def _add_new_neurons(self) -> None:
        """Add new neurons to places where the error is above
        the growing threshold. Begin with the neuron with the largest
        error.
        """
        sorted_indices = np.flip(
            np.argsort(list(dict(self.som_.nodes.data("error")).values()))
        )
        for i in sorted_indices:
            node = list(dict(self.som_.nodes))[i]
            node_degree = nx.degree(self.som_, node)
            if self.som_.nodes[node]["error"] > self.growing_threshold_:
                if node_degree == 1:
                    new_node, new_weight = self._insert_neuron_3p(node)
                elif node_degree == 2:
                    new_node, new_weight = self._insert_neuron_2p(node)
                elif node_degree == 3:
                    new_node, new_weight = self._insert_neuron_1p(node)
                else:
                    continue

                self._add_node_to_graph(node=new_node, weight=new_weight)

            else:
                break

    def _insert_neuron_1p(
        self, node: tuple[int, int]
    ) -> tuple[tuple[int, int], np.ndarray]:
        """Add neuron to the only free position.
        The available positions are:
        - (x_i, y_i + 1)
        - (x_i, y_i - 1)
        - (x_i + 1, y_i)
        - (x_i - 1, y_i)
        """
        node_x, node_y = node
        nbrs = self.som_.adj[node]
        for p1_candidate in [
            (node_x, node_y + 1),
            (node_x, node_y - 1),
            (node_x + 1, node_y),
            (node_x - 1, node_y),
        ]:
            if p1_candidate not in nbrs:
                p1 = p1_candidate
                nb_1 = (2 * node_x - p1[0], 2 * node_y - p1[1])
                new_weight = (
                    2 * self.som_.nodes[node]["weight"]
                    - self.som_.nodes[nb_1]["weight"]
                )

        return p1, new_weight

    def _insert_neuron_2p(
        self, bo: tuple[int, int]
    ) -> tuple[tuple[int, int], np.ndarray]:
        """Add new neuron to the direction with the larger error.

        Case (a):
         o--nb1--nb4
         |   |
        nb2--bo--p1
         |   |
        nb3  p2
        The position P1 is preferable if E(NB4) > E(NB3),
        otherwise P2 is the choice.

        Case (b):
         o--nb1
         |   |
        nb2--bo--p1
             |
             p2
        When there is no neuron adjacent to P1 and P2 (Fig. 3.b), the
        preferable position is P1 if E(NB1) > E(NB2),otherwise a new
        neuron will be added to P2.

        Case (c):
        For the case that the boundary neuron (BO) is not at the corner
        of the grid and there is no neuron adjacent to the available
        positions the preferable position is decided randomly.
        """
        nbr1, nbr2 = self.som_.adj[bo]
        (nbr1_x, nbr1_y), (nbr2_x, nbr2_y) = nbr1, nbr2
        error_nbr1 = self.som_.nodes[nbr1]["error"]
        error_nbr2 = self.som_.nodes[nbr2]["error"]
        bo_x, bo_y = bo
        # corner_neighbor_positions = {
        #     (bo_x + 1, bo_y + 1),
        #     (bo_x + 1, bo_y - 1),
        #     (bo_x - 1, bo_y + 1),
        #     (bo_x - 1, bo_y - 1),
        # }

        # corner_neighbors = list(
        #     corner_neighbor_positions.intersection(set(self.som_.neighbors(nbr1)))
        # )

        # # Case (a)
        # if len(corner_neighbors) == 1:
        #     nbr3 = corner_neighbors[0]
        #     error_nbr3 = self.som_.nodes[nbr3]["error"]

        # else:
        #     nbr3, nbr4 = corner_neighbors[:2]
        #     error_nbr3 = self.som_.nodes[nbr3]["error"]
        #     error_nbr4 = self.som_.nodes[nbr4]["error"]

        # Case (b):
        if error_nbr1 > error_nbr2:
            new_node = (2 * bo_x - nbr2_x, 2 * bo_y - nbr2_y)
            new_weight = (
                2 * self.som_.nodes[bo]["weight"] - self.som_.nodes[nbr2]["weight"]
            )
        else:
            new_node = (2 * bo_x - nbr1_x, 2 * bo_y - nbr1_y)
            new_weight = (
                2 * self.som_.nodes[bo]["weight"] - self.som_.nodes[nbr1]["weight"]
            )

        #  Case (c): Two opposite neighbors
        if nbr1_x == nbr2_x or nbr1_y == nbr2_y:
            if nbr1_x == nbr2_x:
                new_node = (bo_x + 1, bo_y)
                new_weight = (
                    2 * self.som_.nodes[bo]["weight"] - self.som_.nodes[nbr2]["weight"]
                )
            else:
                new_node = (bo_x, bo_y + 1)
                new_weight = (
                    2 * self.som_.nodes[bo]["weight"] - self.som_.nodes[nbr1]["weight"]
                )

        return new_node, new_weight

    def _insert_neuron_3p(
        self, bo: tuple[int, int]
    ) -> tuple[tuple[int, int], np.ndarray]:
        """If the boundary neuron (BO) has three available positions (P1, P2
        and P3), the accumulative error of surrounding neurons should be
        considered according to the insertion rule.

        Case (a):
        nb2  P2
         |    |
        -nb1--BO--P1
         |    |
        nb3  P3

        P1 should be selected if E(NB1) > E(NB2, NB3), otherwise a decision
        should be made just between P2 and P3. The preferable position is P2 if
        E(NB2) > E(NB3), otherwise a new neuron will be added to P3.

        Case (b):
         nb2  P2
          |   |
        -nb1--BO--P1
              |
              P3
        If there is just one neuron adjacent to an available position
        P1 or P2 can be selected for insertion according to the
        same rule as before.

        Case (c):
              P2
              |
        -nb1--BO--P1
              |
              P3
        For the case which there is no neuron adjacent to the
        available positions the position P1 is preferable
        """

        bo_x, bo_y = bo
        corner_neighbor_positions = {
            (bo_x + 1, bo_y + 1),
            (bo_x + 1, bo_y - 1),
            (bo_x - 1, bo_y + 1),
            (bo_x - 1, bo_y - 1),
        }

        nb_1 = list(self.som_.neighbors(bo))[0]
        corner_neighbors = list(
            corner_neighbor_positions.intersection(set(self.som_.neighbors(nb_1)))
        )

        if len(corner_neighbors) == 0:
            new_node, new_weight = self._3p_case_c(nb_1, bo)
        elif len(corner_neighbors) == 1:
            nb_2 = corner_neighbors[0]
            new_node, new_weight = self._3p_case_b(nb_1, bo, nb_2)
        else:
            nb_2 = corner_neighbors[0]
            nb_3 = corner_neighbors[1]
            new_node, new_weight = self._3p_case_a(nb_1, bo, nb_2, nb_3)

        return new_node, new_weight

    def _3p_case_a(
        self,
        nb_1: tuple[int, int],
        bo: tuple[int, int],
        nb_2: tuple[int, int],
        nb_3: tuple[int, int],
    ) -> tuple[tuple[int, int], np.ndarray]:
        if (
            self.som_.nodes[nb_1]["error"] > self.som_.nodes[nb_2]["error"]
            and self.som_.nodes[nb_1]["error"] > self.som_.nodes[nb_3]["error"]
        ):
            new_node, new_weight = self._3p_case_c(nb_1, bo)
        elif self.som_.nodes[nb_2]["error"] > self.som_.nodes[nb_3]["error"]:
            new_node, new_weight = self._3p_case_b(nb_1, bo, nb_2)
        else:
            new_node, new_weight = self._3p_case_b(nb_1, bo, nb_2)

        return new_node, new_weight

    def _3p_case_b(
        self, nb_1: tuple[int, int], bo: tuple[int, int], nb_2: tuple[int, int]
    ) -> tuple[tuple[int, int], np.ndarray]:
        if self.som_.nodes[nb_1]["error"] > self.som_.nodes[nb_2]["error"]:
            new_node, new_weight = self._3p_case_c(nb_1, bo)
        else:
            new_node = (
                nb_2[0] + bo[0] - nb_1[0],
                nb_2[1] + bo[1] - nb_1[1],
            )

            new_weight = (
                (2 * self.som_.nodes[bo]["weight"] - self.som_.nodes[nb_1]["weight"])
                + self.som_.nodes[nb_2]["weight"]
            ) / 2

        return new_node, new_weight

    def _3p_case_c(
        self, neighbor: tuple[int, int], node: tuple[int, int]
    ) -> tuple[tuple[int, int], np.ndarray]:
        new_node = (2 * node[0] - neighbor[0], 2 * node[1] - neighbor[1])
        new_weight = (
            2 * self.som_.nodes[node]["weight"] - self.som_.nodes[neighbor]["weight"]
        )
        return new_node, new_weight

    def _add_node_to_graph(self, node: tuple[int, int], weight: np.ndarray) -> None:
        self.som_.add_node(node)
        self.som_.nodes[node]["weight"] = weight
        self.som_.nodes[node]["error"] = 0
        self.som_.nodes[node]["epoch_created"] = self._current_epoch
        self._add_new_connections(node)

    def _add_new_connections(self, node: tuple[int, int]) -> None:
        """Given a node (x, y), add new connections to the neighbors of the
        node, if exist."""
        node_x, node_y = node
        for nbr in [
            (node_x, node_y + 1),
            (node_x, node_y - 1),
            (node_x - 1, node_y),
            (node_x + 1, node_y),
        ]:
            if nbr in self.som_.nodes:
                self.som_.add_edge(node, nbr)

    def _sigma(self) -> float:
        """Return the neighborhood bandwidth for each epoch.
        If no sigma is given, the starting bandwidth is set to
        0.2 * sqrt(n_neurons) and the ending bandwidth is set to
        max(0,7, 0.05 * sqrt(n_neurons)) where n_neurons is the
        number of neurons in the graph in the current epoch.

        Returns:
            float: The neighborhood bandwidth for each epoch.
        """
        epoch = self._current_epoch
        n_neurons = self.som_.number_of_nodes()
        if self.sigma_start is None:
            sigma_start = 0.2 * np.sqrt(n_neurons)
        else:
            sigma_start = self.sigma_start

        if self.sigma_end is None:
            sigma_end = max(0.7, 0.05 * np.sqrt(n_neurons))
        else:
            sigma_end = self.sigma_end

        if self._training_phase == "coarse":
            if self.decay_function == "linear":
                sigma = sigma_start * (
                    1 - (1 / self.coarse_training_frac * epoch / self.n_epochs_max)
                ) + sigma_end * (epoch / self.n_epochs_max)

            elif self.decay_function == "exponential":
                sigma = sigma_start * np.exp(
                    (1 / self.n_epochs_max * (log(sigma_end) - log(sigma_start)))
                    * epoch
                    / self.coarse_training_frac
                )
        else:
            sigma = sigma_end

        return sigma

    def calculate_quantization_error(self, X: npt.ArrayLike) -> float:
        """Return the average distance from each sample to the nearest
        prototype.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Data to quantize.

        Returns
        -------
        error : float
            Average distance from each sample to the nearest prototype.
        """
        check_is_fitted(self)
        X = check_array(X)
        winners = self._get_winning_neurons(X, n_bmu=1)
        error = np.mean(np.linalg.norm(self.weights_[winners] - X, axis=1))
        return error

    def _topographic_error_func(self, X: npt.ArrayLike) -> float:
        """Return the topographic error of the training data.

        The topographic error is a measure for the topology preservation of
        the map.

        For each sample we get the two best matching units. If the BMU are
        connected on the grid, there is no error. If the distance is
        larger an error occurred. The total error is the number
        of single errors divided by the number of samples.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Data to show the SOM.

        Returns
        -------
        topographic error : float
            Fraction of samples with topographic errors over all samples.
        """
        bmu_indices = self._get_winning_neurons(X, n_bmu=2).T
        errors = 0
        for bmu_1, bmu_2 in bmu_indices:
            dist = self._distance_matrix[bmu_1, bmu_2]
            if dist > 1:
                errors += 1

        return errors / X.shape[0]

"""
This class handles the core SOM functionality.
"""

import copy
import sys
from math import exp, log, sqrt, pi
from numbers import Integral
from typing import Any

# from matplotlib import pyplot as plt
# import matplotlib

try:
    import networkx as nx
    import numba as nb
    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    import scipy.spatial.distance
    import seaborn.objects as so
    from sklearn.base import BaseEstimator, clone
    from sklearn.decomposition import SparseCoder
    from sklearn.metrics import pairwise_distances

    # from line_profiler import profile
    from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize
    from sklearn.utils import check_array, check_random_state
    from sklearn.utils._param_validation import Interval, StrOptions
    from sklearn.utils.validation import check_is_fitted
    from tqdm import tqdm
except ImportError as e:
    print(e)
    sys.exit()


# pylint:  disable= attribute-defined-outside-init
class BaseSom(BaseEstimator):

    def __init__(
        self,
        n_iter: int = 200,
        convergence_iter: int = 1,
        spreading_factor: float = 0.5,
        sigma_start: float | None = None,
        sigma_end: float | None = None,
        vertical_growth: bool = False,
        decay_function: str = "exponential",
        learning_rate: float = 0.02,
        verbose=False,
        coarse_training_frac: float = 0.5,
        random_state: Any = None,
        convergence_treshold: float = 10**-5,
        max_neurons: int = 100,
        metric: str = "euclidean",
        threshold_method: str = "se",
        growth_criterion: str = "quantization_error",
        min_samples_vertical_growth: int = 100,
        n_jobs: int = 1,
    ) -> None:
        self.spreading_factor = spreading_factor
        self.n_iter = n_iter
        self.convergence_iter = convergence_iter
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.decay_function = decay_function
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.coarse_training_frac = coarse_training_frac
        self.random_state = random_state
        self.convergence_treshold = convergence_treshold
        self.max_neurons = max_neurons
        self.metric = metric
        self.threshold_method = threshold_method
        self.growth_criterion = growth_criterion
        self.min_samples_vertical_growth = min_samples_vertical_growth
        self.vertical_growth = vertical_growth
        self.n_jobs = n_jobs

    _parameter_constraints = {
        "n_iter": [Interval(Integral, 1, None, closed="left")],
        "max_neurons": [Interval(Integral, 4, None, closed="left")],
        "decay_function": [StrOptions({"exponential", "linear"})],
    }

    def fit(self, X: npt.ArrayLike, y: None | npt.ArrayLike = None):
        """Train SOM on training data.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Training data.

        y : array_like of shape (n_samples), optional
            Class labels of the samples.

        Returns
        -------
        self : DBGSOM
            Trained estimator
        """
        # Initialization
        X, y = self._check_input_data(X, y)
        # self._fit(X, y)
        if y is not None:
            classes, y = np.unique(y, return_inverse=True)
            self.classes_ = np.array(classes)
        self.random_state_ = check_random_state(self.random_state)
        self._initialize_som(X)

        # Horizontal growing phase
        self._grow_som(X, y)
        # self.rep = self._calculate_rep(X)
        self.topographic_error_ = self._calculate_topographic_error(X)
        self.quantization_error_ = self.calculate_quantization_error(X)
        self.n_features_in_ = X.shape[1]
        self._write_node_statistics(X)
        self._delete_dead_neurons_from_graph(X)
        self._label_prototypes(X, y)

        # Vertical growing phase
        if self.vertical_growth:
            self._grow_vertical(X, y)

        self._fit(X)
        # self.labels_ = self.predict(X)
        self.n_iter_ = self._current_epoch

        return self

    def _check_input_data(self, X, y):
        raise NotImplementedError

    def _fit(self, X):
        # For VQ Subclass specific code
        pass

    def predict(self, X):
        raise NotImplementedError

    def _check_arguments(self):
        if self.decay_function not in ["linear", "exponential"]:
            raise ValueError(
                "Decay function not supported. Must be 'linear' or 'exponential'."
            )
        if self.threshold_method not in ["se", "classical"]:
            raise ValueError(
                "threshold_method not supported. Must be 'se' or 'classical'."
            )
        if self.growth_criterion not in ["quantization_error", "entropy"]:
            raise ValueError(
                "growth_criterion not supported. Must be 'quantization_error' or 'entropy'."
            )

    def _grow_vertical(self, X: npt.ArrayLike, y: None | npt.ArrayLike = None) -> None:
        """
        Triggers vertical growth in the SOM by creating new instances of the DBGSOM
        class and fitting them with filtered data.
        """
        # todo: refactor in sub classes
        self.vertical_growing_threshold_ = 1.5 * self.growing_threshold_
        _, winners = self._get_winning_neurons(X, n_bmu=1)
        relevant_nodes = [
            node
            for (node, error) in enumerate(self.som_.nodes(data="error"))
            if error > self.vertical_growing_threshold_
        ]
        for node in relevant_nodes:
            new_som = clone(self)
            X_filtered = X[winners == node]
            if y is not None:
                y_filtered = y[winners == node]
            else:
                y_filtered = None
            if X_filtered.shape[0] > self.min_samples_vertical_growth:
                new_som.fit(X_filtered, y_filtered)
                self.som_.nodes[node]["som"] = new_som

    def _calculate_node_statistics(
        self, X: npt.ArrayLike
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Write the following statistics as attributes to the graph:

        1. Local density: Use a Gaussian kernel to estimate the local density around
        each prototype. Use the average distance from all prototype to their neighbors
        as bandwith sigma.

        2. Hit count: How many samples each prototype represents.

        3. average distance: average distance from each prototype to their neighbors.
        used for plotting the u matrix"""
        distances, winners = self._get_winning_neurons(X, n_bmu=1)
        average_distances = self._get_u_matrix()
        sigma = average_distances.mean()
        densities = np.zeros((len(self.neurons_)))
        hit_counts = np.zeros((len(self.neurons_)))
        for winner in np.unique(winners):
            samples = X[winners == winner]
            distances_per_neuron = distances[winners == winner]
            if len(distances_per_neuron) > 0:
                local_density = np.mean(
                    (np.exp(-(distances_per_neuron**2) / (2 * sigma**2)))
                    / (sigma * sqrt(2 * pi))
                )
            else:
                local_density = 0
            densities[winner] = local_density
            hit_counts[winner] = len(samples)
        return average_distances, densities, hit_counts

    def _write_node_statistics(self, X: npt.ArrayLike) -> None:
        average_distances, densities, hit_counts = self._calculate_node_statistics(X)

        for density, hit_count, average_distance, node in zip(
            densities, hit_counts, average_distances, self.som_.nodes
        ):
            self.som_.nodes[node]["density"] = density
            self.som_.nodes[node]["hit_count"] = hit_count
            self.som_.nodes[node]["average_distance"] = average_distance

    def _delete_dead_neurons_from_graph(self, X: npt.ArrayLike) -> None:
        """Delete all neurons which represent zero samples from the training set."""
        som_copy = copy.deepcopy(self.som_)
        dead_neurons = [
            node for node in self.som_.nodes if self.som_.nodes[node]["hit_count"] == 0
        ]
        for node in dead_neurons:
            som_copy.remove_node(node)
        self.som_ = som_copy

        self.neurons_ = list(self.som_.nodes)
        self.weights_ = self._extract_values_from_graph("weight")
        self._distance_matrix = nx.floyd_warshall_numpy(self.som_)

    def _extract_values_from_graph(self, attribute: str) -> np.ndarray:
        """Return an array with some given attribute of the nodes."""
        return np.array([data[attribute] for _, data in self.som_.nodes.data()])

    def transform(self, X: npt.ArrayLike, y=None) -> np.ndarray:
        """Calculate a non negative least squares mixture model of prototypes that
        approximate each sample.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data to transform.

        y : Ignored.
            Not used, present here for API consistency by convention.

        Returns
        -------
        coefficients : np.ndarray of shape (n_samples, n_protoypes)
            Coefficients of the linear regression model.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=[np.float64, np.float32])
        transformer = SparseCoder(
            dictionary=normalize(self.weights_),
            n_jobs=self.n_jobs,
            positive_code=True,
            transform_alpha=0,
            transform_algorithm="lasso_lars",
        )
        coefs = transformer.transform(normalize(X))
        return coefs

    def plot(self, color: None | str = None, palette="magma_r", pointsize=None) -> None:
        """Plot the neurons.

        Parameters
        ----------
        color, pointsize : {None, "label", "epoch_created", "error", "average_distance",
        "density", "hit_count"}, default = None
            Attribute which is represented as color.

            "label" : Label of the prototype when trained supervised.

            "epoch_created" : When the neuron was created.

            "error" : Quantization error of each neuron.

            "average_distance" : Average distance to neighbor neurons in
            the input space. Creates a U-Matrix.

            "density" : estimated local density around the prototype

            "hit_count" : How many samples the prototype represents

        palette : matplotlib colormap/seaborn palette, default = "magma_r"
            Name of seaborn palette to color code the values of attribute
        """
        data = pd.DataFrame(dict(self.som_.nodes)).T.set_index(
            np.arange(len(self.som_.nodes))
        )

        data["label_index"] = pd.to_numeric(data["label"])
        data["label"] = self.classes_[data["label_index"]]

        data["epoch_created"] = pd.to_numeric(data["epoch_created"])
        data["error"] = pd.to_numeric(data["error"])
        data["density"] = pd.to_numeric(data["density"])
        data["hit_count"] = pd.to_numeric(data["hit_count"])
        data["average_distance"] = pd.to_numeric(data["average_distance"])
        coordinates = pd.DataFrame(np.array(self.neurons_), columns=["x", "y"])
        dots = pd.concat([coordinates, data], axis=1)
        # matplotlib.use("nbAgg")
        # f = plt.figure()
        so.Plot(dots, x="x", y="y", color=color, pointsize=pointsize).add(
            so.Dot()
        ).scale(color=palette).label(
            x="", y=""
        ).show()  # .on(f)
        # f
        # f.show()
        # plt.show()

    def _get_u_matrix(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Calculate the average distance from each neuron to its neighbors in the
        input space."""

        g = self.som_
        node_weights = np.array([g.nodes[node]["weight"] for node in g.nodes])
        neighbor_weights = np.array(
            [
                g.nodes[neighbor]["weight"]
                for neighbors in g.adj.values()
                for neighbor in neighbors
            ]
        )
        distances = scipy.spatial.distance.cdist(node_weights, neighbor_weights).mean(
            axis=1
        )

        return distances

    def _calculate_rep(self, X: npt.ArrayLike) -> None:
        """Return the resemble entropy parameter.

        1. Calculate histogram of components of each sample.
        2. Calculate entropy of each sample from histogram
        3. Save minimum and maximum rep for all classes

        Use 20 bins as default"""

        hists = []
        for sample in X:
            hists.append(np.histogram(sample, bins=20)[0])

    def _initialize_som(self, data: npt.NDArray) -> None:
        """First training phase.

        Calculate growing threshold according to the argument. Create
        a graph containing the first four neurons in a square with
        init vectors.
        """
        self._current_epoch = 0
        self.converged_ = False
        self._training_phase = "coarse"
        self.growing_threshold_ = self._calculate_growing_threshold(data)

        self.som_ = self._create_som(data)
        self._distance_matrix = nx.floyd_warshall_numpy(self.som_)
        self.weights_ = self._extract_values_from_graph("weight")
        self.neurons_ = list(self.som_.nodes)

    def _calculate_growing_threshold(self, data: npt.NDArray) -> float:
        if self.growth_criterion == "entropy":
            growing_threshold = self.spreading_factor
        else:
            if self.threshold_method == "classical":
                n_dim = data.shape[1]
                growing_threshold = -n_dim * log(self.spreading_factor)

            elif self.threshold_method == "se":
                std_data = np.std(data, axis=0, ddof=1)
                growing_threshold = float(
                    150 * -log(self.spreading_factor) * np.linalg.norm(std_data)
                )

        return growing_threshold

    def _grow_som(self, data: npt.NDArray, y: np.ndarray) -> None:
        """Second training phase"""
        for current_epoch in tqdm(
            iterable=range(self.n_iter),
            disable=not self.verbose,
            unit=" epochs",
        ):
            self._current_epoch = current_epoch
            if current_epoch > self.coarse_training_frac * self.n_iter:
                self._training_phase = "fine"
            self.weights_ = self._extract_values_from_graph("weight")
            # check if new neurons were inserted
            if len(self.som_.nodes) > len(self.neurons_) or current_epoch == 0:
                self.neurons_ = list(self.som_.nodes)
                self._distance_matrix = nx.floyd_warshall_numpy(self.som_)

            distances, winners = self._get_winning_neurons(data, n_bmu=1)
            sample_weights = self._calculate_exp_similarity(distances, data)

            self._update_weights(sample_weights, winners, data)
            self._write_accumulative_error(winners, y, distances)
            if self.converged_ and self._training_phase == "fine":
                break

            if (
                self._training_phase == "coarse"
                and len(self.neurons_) < self.max_neurons
                and current_epoch % self.convergence_iter == self.convergence_iter - 1
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

    def _get_winning_neurons(
        self, data: npt.NDArray, n_bmu: int
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Calculate distances from each neuron to each sample.

        Return distances and index of the winning neuron(s) or
        best matching units(s) for each sample.
        """
        weights = self.weights_
        nn_tree = NearestNeighbors(n_neighbors=n_bmu)
        nn_tree.fit(weights)
        result = nn_tree.kneighbors(data)
        distances = result[0]
        winners = result[1].T[0:n_bmu].T
        if n_bmu == 1:
            winners = winners.reshape(-1)
            distances = distances.reshape(-1)

        return distances, winners

    def _label_prototypes(self, X, y) -> None:
        raise NotImplementedError

    # @profile
    def _update_weights(
        self, sample_weights: np.ndarray, winners: np.ndarray, data: npt.NDArray
    ) -> None:
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
        # see https://stackoverflow.com/questions/75423927/what-is-the-fastest-way
        # -to-select-multiple-elements-from-a-numpy-array/75424204#75424204

        index = np.argsort(winners)
        groups, offsets = np.unique(winners[index], return_index=True)
        voronoi_set_centers = numba_voronoi_set_centers(
            kernel=sample_weights,
            data=data,
            shape=self.weights_.shape,
            groups=groups,
            offsets=offsets,
            index=index,
        )

        # step 2
        neuron_activations = np.zeros(shape=len(self.neurons_))
        winners, winner_counts = np.unique(winners, return_counts=True)
        for winner, count in zip(winners, winner_counts):
            neuron_activations[winner] = count

        # Step 3
        gaussian_kernel = self._calculate_gaussian_neighborhood()

        # Step 4
        intermediate_calculation = (
            gaussian_kernel[:, :, np.newaxis] * neuron_activations[:, np.newaxis]
        )
        new_weights = np.sum(
            voronoi_set_centers * intermediate_calculation,
            axis=1,
        ) / np.sum(intermediate_calculation, axis=1)

        # Step 5
        new_weights_dict = dict(zip(self.neurons_, new_weights))
        change = np.linalg.norm(self.weights_ - new_weights, axis=1)
        change_total = np.sum(change)
        if change_total < self.convergence_treshold:
            self.converged_ = True
        nx.set_node_attributes(G=self.som_, values=new_weights_dict, name="weight")

    def _calculate_gaussian_neighborhood(self) -> np.ndarray:
        """Calculate the gaussian neighborhood function for all neuron
        pairs using the distance matrix."""
        sigma = self._calculate_current_sigma()
        h = np.exp(-(self._distance_matrix**2 / (2 * sigma**2)))

        return h

    def _calculate_exp_similarity(
        self, distances: np.ndarray, data: np.ndarray
    ) -> np.ndarray:
        """Calculate the weight of each sample by calculating a exponential kernel
        for the distance between the sample and the bmu."""
        gamma = np.var(data, axis=0).sum() ** -1
        kernel = 1 - (1 - np.exp(-gamma * distances**2)) ** 0.5
        return kernel

    # @profile
    def _write_accumulative_error(
        self, winners: np.ndarray, y: np.ndarray, distances: np.ndarray
    ) -> None:
        """Get the quantization error for each neuron
        and save it as "error" attribute of each node.
        """
        if self.growth_criterion == "entropy":
            for winner_index, neuron in enumerate(self.neurons_):
                counts = np.bincount(y[winners == winner_index])
                error = scipy.stats.entropy(counts, base=2)
                self.som_.nodes[neuron]["error"] = error

        else:
            errors = numba_quantization_error(
                winners,
                length=self.weights_.shape[0],
                distances=distances,
            )
            for i, error in enumerate(errors):
                neuron = self.neurons_[i]
                self.som_.nodes[neuron]["error"] = error

    def _distribute_errors(self) -> None:
        """
        Distributes the error values of neurons in the SOM which are not boundary
        neurons to their neighboring boundary neurons. This distribution is done
        when the error value of a neuron is greater than a predefined threshold.
        """
        for node, neighbors in self.som_.adj.items():
            is_boundary = len(neighbors) < 4
            node_error = self.som_.nodes[node]["error"]

            if not is_boundary and node_error > self.growing_threshold_:
                boundary_neighbors = [
                    neighbor
                    for neighbor in neighbors.keys()
                    if len(self.som_.adj[neighbor]) < 4
                ]
                n_boundary_neighbors = len(boundary_neighbors)

                for neighbor in boundary_neighbors:
                    self.som_.nodes[neighbor]["error"] += (
                        0.5 * node_error / n_boundary_neighbors
                    )

                self.som_.nodes[node]["error"] /= 2

    def _add_new_neurons(self) -> None:
        """Add new neurons to places where the error is above
        the growing threshold. Begin with the neuron with the largest
        error.
        """
        error_values = self._extract_values_from_graph("error")
        sorted_indices = np.argsort(-error_values)

        for i in sorted_indices:
            node = list(self.som_.nodes)[i]
            node_degree = nx.degree(self.som_, node)
            degree_functions = {
                1: self._insert_neuron_3p,
                2: self._insert_neuron_2p,
                3: self._insert_neuron_1p,
            }
            if error_values[i] > self.growing_threshold_ and node_degree < 4:
                if node_degree in degree_functions:
                    new_node, new_weight = degree_functions[node_degree](node)
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
        possible_positions = [
            (node_x, node_y + 1),
            (node_x, node_y - 1),
            (node_x + 1, node_y),
            (node_x - 1, node_y),
        ]
        for new_position_candidate in possible_positions:
            if new_position_candidate not in nbrs:
                new_position = new_position_candidate
                neighbor_position = (
                    2 * node_x - new_position[0],
                    2 * node_y - new_position[1],
                )
                new_weight = (
                    2 * self.som_.nodes[node]["weight"]
                    - self.som_.nodes[neighbor_position]["weight"]
                )

        return new_position, new_weight

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
        error_nb_1 = self.som_.nodes[nb_1]["error"]
        error_nb_2 = self.som_.nodes[nb_2]["error"]
        error_nb_3 = self.som_.nodes[nb_3]["error"]

        if error_nb_1 > error_nb_2 and error_nb_1 > error_nb_3:
            new_node, new_weight = self._3p_case_c(nb_1, bo)
        elif error_nb_2 > error_nb_3:
            new_node, new_weight = self._3p_case_b(nb_1, bo, nb_2)
        else:
            new_node, new_weight = self._3p_case_b(nb_1, bo, nb_3)

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
        attributes = {
            "weight": weight,
            "error": 0,
            "epoch_created": self._current_epoch,
        }
        self.som_.nodes[node].update(attributes)
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

    def _calculate_current_sigma(self) -> float:
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
            sigma_start = 0.2 * sqrt(n_neurons)
        else:
            sigma_start = self.sigma_start

        if self.sigma_end is None:
            sigma_end = max(0.7, 0.05 * sqrt(n_neurons))
        else:
            sigma_end = self.sigma_end

        if self._training_phase == "coarse":
            if self.decay_function == "linear":
                decay_function = linear_decay

            elif self.decay_function == "exponential":
                decay_function = exponential_decay

            sigma = decay_function(
                sigma_end=sigma_end,
                sigma_start=sigma_start,
                max_iter=self.n_iter,
                current_iter=epoch / self.coarse_training_frac,
                learning_rate=self.learning_rate,
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
        distances, _ = self._get_winning_neurons(X, n_bmu=1)
        error = float(np.mean(distances))
        return error

    def _calculate_topographic_error(self, X: npt.ArrayLike) -> float:
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
        _, bmu_indices = self._get_winning_neurons(X, n_bmu=2)
        euclid_dist_matrix = euclidean_distances(self.neurons_)
        topographic_error = 0
        for node in bmu_indices:
            # distance = int(distance_matrix[node[0], node[1]])
            distance = euclid_dist_matrix[node[0], node[1]]
            topographic_error += 1 if distance > 1.5 else 0

        return topographic_error / X.shape[0]

    def topographic_function(self, X: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        X = check_array(X)
        self._delaunay_maxtrix = self._calculate_delaunay_triangulation(X)
        self.euclid_dist_matrix = euclidean_distances(self.neurons_)
        self.manhattan_dist_matrix = manhattan_distances(self.neurons_)
        self.max_dist_matrix = pairwise_distances(self.neurons_, metric="chebyshev")
        max_dist = int(self.max_dist_matrix.max())
        k_positive = np.arange(max_dist)
        k_negative = np.arange(max_dist)
        for k in range(max_dist):
            k_positive[k] = self.phi(k)
            k_negative[k] = self.phi(-k)

        return (
            k_positive / len(self.neurons_),
            k_negative / len(self.neurons_),
        )

    def phi(self, k: int) -> int:
        if k > 0:
            return np.count_nonzero(
                (self.max_dist_matrix > k) & (self._delaunay_maxtrix == 1)
            )
        elif k < 0:
            return np.count_nonzero(
                (self.euclid_dist_matrix == 1) & (self._delaunay_maxtrix > -k)
            )
        else:
            return self.phi(-1) + self.phi(1)

    def _calculate_delaunay_triangulation(self, X) -> np.ndarray:
        """Calculate the Delaunay triangulation distance matrix."""
        _, bmu_indices = self._get_winning_neurons(X, n_bmu=2)

        n_neurons = self.som_.number_of_nodes()
        connectivity_matrix = np.zeros(shape=(n_neurons, n_neurons))
        for node in bmu_indices:
            connectivity_matrix[node[0], node[1]] = 1
            connectivity_matrix[node[1], node[0]] = 1

        delaunay_triangulation_graph = nx.from_numpy_array(connectivity_matrix)
        distance_matrix = nx.floyd_warshall_numpy(delaunay_triangulation_graph)

        return distance_matrix


def linear_decay(
    sigma_start: float,
    sigma_end: float,
    max_iter: int,
    current_iter: float,
    learning_rate=None,
) -> float:
    """Linear decay between sigma_start and sigma_end over t training iterations."""
    ratio = current_iter / max_iter
    sigma = sigma_start * (1 - ratio) + sigma_end * ratio

    return sigma


def exponential_decay(
    sigma_start: float,
    sigma_end: float,
    max_iter: int,
    current_iter: float,
    learning_rate: float,
) -> float:
    """Exponential decay between sigma_start and sigma_end with a given learning rate."""
    sigma = sigma_end + (sigma_start - sigma_end) * exp(-learning_rate * current_iter)

    return sigma


@nb.njit(
    parallel=True,
    fastmath=True,
)
def numba_voronoi_set_centers(
    kernel,
    data: npt.NDArray,
    shape: tuple,
    groups: npt.NDArray,
    offsets: npt.NDArray,
    index: npt.NDArray,
) -> np.ndarray:
    """
    Calculates the centers of the Voronoi regions based on the winners and data arrays.
    """

    voronoi_set_centers = np.zeros(shape=shape)
    for i in nb.prange(groups.size):
        group_start = offsets[i]
        group_end = offsets[i + 1] if i + 1 < groups.size else index.size
        group_index = index[group_start:group_end]
        samples = data[group_index]
        weights = kernel[group_index]
        for j in nb.prange(samples.shape[1]):
            mean_samples = np.average(samples[:, j], weights=weights)
            voronoi_set_centers[i, j] = mean_samples

    return voronoi_set_centers


@nb.njit(
    fastmath=True,
    parallel=True,
)
def numba_quantization_error(
    winners: npt.NDArray, length: int, distances: npt.NDArray
) -> np.ndarray:
    """
    Calculate the quantization error for a given set of winners, distances, and length.
    """
    errors = np.zeros(shape=length)
    for i in nb.prange(len(winners)):
        winner = winners[i]
        distance = distances[i]
        errors[winner] += distance
    return errors

from math import log
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist
from tqdm import tqdm


class DBGSOM:
    """A Directed Batch Growing Self-Organizing Map.

    Parameters
    ----------
    sf : float (default = 0.4)
        Spreading factor to calculate the treshold for neuron insertion.
        0 <= sf <= 1.

    n_epochs : int (default = 30)
        Number of training epochs.

    sigma_start, sigma_end : {None, numeric}, default: None
        Start and end values for the neighborhood bandwidth.
        If None, it is calculated dynamically in each epoch.

    decay_function : {'exponential', 'linear'}, default: 'exponential'
        Decay function to use for neighborhood bandwith sigma.

    coarse_training_frac : float (default = 1)
        Fraction of training epochs to use for coarse training.
        In coarse training, the neighborhood bandwidth is decreased from
        sigma_start to sigma_end and the network grows according to the
        growing rules. In fine training, the bandwidth is constant at
        sigma_end and no new neurons are added.

    random_state : any (optional, default = None)
        Random state for weight initialization.

    convergence_treshold : float (default = 10 ** -10)
        If the sum of all weight changes is smaller than the threshold,
        convergence is assumed and the training is stopped.
    """

    def __init__(
        self,
        n_epochs: int = 30,
        sf: float = 0.4,
        sigma_start: float | None = None,
        sigma_end: float | None = None,
        decay_function: str = "exponential",
        coarse_training_frac: float = 1,
        random_state: Any = None,
        convergence_treshold: float = 10**-10,
    ) -> None:
        self.SF = sf
        self.N_EPOCHS = n_epochs
        self.SIGMA_START = sigma_start
        self.SIGMA_END = sigma_end
        self.DECAY_FUNCTION = decay_function
        self.COARSE_TRAINING_FRAC = coarse_training_frac
        self.RANDOM_STATE = random_state
        self.training_phase = "coarse"
        self.rng = np.random.default_rng(seed=self.RANDOM_STATE)
        self.convergence_treshold = convergence_treshold
        self.converged = False

        # Only for python style guide. These are created in _initialization()
        # at training time
        self.current_epoch = 0
        self.GROWING_TRESHOLD = None
        self.distance_matrix: np.ndarray | None = None
        self.som: nx.Graph = None
        self.weights = None
        self.neurons: list[tuple[int, int]] = None

    def fit(self, X) -> None:
        """Train SOM on training data.

        Parameters
        ----------
        data : array_like of shape (n_samples, n_features)
            Training data.
        """
        self._initialization(X)
        self._grow(X)

    def predict(self, X) -> np.ndarray:
        """Predict the closest cluster each sample in X belongs to. In the
        vector quantization literature, cluster_centers_ is called the
        code book and each value returned by predict is the index of the
        closest code in the code book."""
        return self._get_winning_neurons(X, n_bmu=1)

    def _initialization(self, data: npt.NDArray) -> None:
        """First training phase.

        Calculate growing threshold as gt = -data_dimensions * log(spreading_factor).
        Create a graph containing the first four neurons in a square with
        init vectors.
        """
        data = data.astype(np.float32)
        # BATCH_SIZE = np.sqrt(len(data))
        # self.N_BATCHES = int(len(data) / BATCH_SIZE)
        self.GROWING_TRESHOLD = -data.shape[1] * log(self.SF)

        self.som = self._create_som(data)
        self.distance_matrix = nx.floyd_warshall_numpy(self.som)
        self.weights = np.array(list(dict(self.som.nodes.data("weight")).values()))
        self.neurons = list(self.som.nodes)

    def _grow(self, data: npt.NDArray) -> None:
        """Second training phase"""
        for i in tqdm(
            iterable=range(self.N_EPOCHS),
            unit=" epochs",
        ):
            self.current_epoch = i + 1
            if self.current_epoch > self.COARSE_TRAINING_FRAC * self.N_EPOCHS:
                self.training_phase = "fine"
            self.weights = np.array(list(dict(self.som.nodes.data("weight")).values()))
            if len(self.som.nodes) > len(self.neurons) or self.current_epoch == 1:
                self.neurons = list(self.som.nodes)
                self._update_distance_matrix()

            winners = self._get_winning_neurons(data, n_bmu=1)
            self._update_weights(winners, data)

            self._write_accumulative_error(winners, data)
            if self.converged:
                print(self.current_epoch)
                break
            if self.current_epoch != self.N_EPOCHS and self.training_phase == "coarse":
                self._distribute_errors()
                self._add_new_neurons()

    def _create_som(self, data: npt.NDArray) -> nx.Graph:
        """Create a graph containing the first four neurons in a square.
        Each neuron has a weight vector randomly chosen from the training
         samples."""
        init_vectors = self.rng.choice(a=data, size=4, replace=False)
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
        distances = cdist(self.weights, data)
        #  Argmin is 10x faster than argsort
        if n_bmu == 1:
            winners = np.argmin(distances, axis=0)
        else:
            winners = np.argsort(distances, axis=0)[:n_bmu]

        return winners

    def _update_distance_matrix(self) -> None:
        """Update distance matrix between neurons.
        Only paths of length =< 3 * sigma + 1 are considered for performance
        reasons.
        """
        n_neurons = len(self.neurons)
        m = np.zeros((n_neurons, n_neurons))
        m.fill(np.inf)
        dist_dict = dict(
            nx.all_pairs_shortest_path_length(self.som, cutoff=3 * self._sigma() + 1)
        )
        for i1, neuron1 in enumerate(self.neurons):
            for i2, neuron2 in enumerate(self.neurons):
                if neuron2 in dist_dict[neuron1].keys():
                    m[i1, i2] = dist_dict[neuron1][neuron2]

        self.distance_matrix = m

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
        voronoi_set_centers = np.zeros_like(self.weights)
        for winner in np.unique(winners):
            voronoi_set_centers[winner] = data[winners == winner].mean(axis=0)

        neuron_activations = np.zeros(shape=len(self.neurons), dtype=np.float32)

        winners, winner_counts = np.unique(winners, return_counts=True)
        for winner, count in zip(winners, winner_counts):
            neuron_activations[winner] = count

        gaussian_kernel = self._gaussian_neighborhood()
        numerator = np.sum(
            voronoi_set_centers
            * neuron_activations[:, np.newaxis]
            * gaussian_kernel[:, :, np.newaxis],
            axis=1,
        )
        denominator = np.sum(
            gaussian_kernel[:, :, np.newaxis] * neuron_activations[:, np.newaxis],
            axis=1,
        )
        new_weights = numerator / denominator

        new_weights_dict = dict(zip(self.neurons, new_weights))
        change = np.linalg.norm(self.weights - new_weights, axis=1)
        change_total = np.sum(change)
        if change_total < self.convergence_treshold:
            self.converged = True
        nx.set_node_attributes(G=self.som, values=new_weights_dict, name="weight")

    def _gaussian_neighborhood(self) -> np.ndarray:
        """Calculate the gaussian neighborhood function for all neuron
        pairs."""
        sigma = self._sigma()
        h = np.exp(-(self.distance_matrix**2 / (2 * sigma**2))).astype(np.float32)

        return h

    def _write_accumulative_error(self, winners: np.ndarray, data: npt.NDArray) -> None:
        """Get the quantization error for each neuron
        and save it as "error" to the graph.
        """
        for winner_index, _ in enumerate(self.neurons):
            samples = data[winners == winner_index]
            dist = np.linalg.norm(self.weights[winner_index] - samples, axis=1)
            error = dist.sum()
            self.som.nodes[self.neurons[winner_index]]["error"] = error

    def _distribute_errors(self) -> None:
        """For each neuron i which is not a boundary neuron and E_i > GT,
        a half value of E_i is equally distributed to the neighboring
        boundary neurons, if exist.
        """
        for node, neighbors in self.som.adj.items():
            if len(neighbors.items()) == 4:
                is_boundary = False
            else:
                is_boundary = True
            node_error = self.som.nodes[node]["error"]

            if not is_boundary and node_error > self.GROWING_TRESHOLD:
                n_boundary_neighbors = 0
                for neighbor in neighbors.keys():
                    if len(self.som.adj[neighbor].items()) < 4:
                        n_boundary_neighbors += 1

                for neighbor in neighbors.keys():
                    if len(self.som.adj[neighbor].items()) < 4:
                        self.som.nodes[neighbor]["error"] += (
                            0.5 * node_error / n_boundary_neighbors
                        )
                self.som.nodes[node]["error"] /= 2

    def _add_new_neurons(self) -> None:
        """Add new neurons to places where the error is above
        the growing threshold.
        """
        sorted_indices = np.flip(
            np.argsort(list(dict(self.som.nodes.data("error")).values()))
        )
        for i in sorted_indices:
            node = list(dict(self.som.nodes))[i]
            if self.som.nodes[node]["error"] > self.GROWING_TRESHOLD:
                if nx.degree(self.som, node) == 1:
                    self._insert_neuron_3p(node)
                elif nx.degree(self.som, node) == 2:
                    self._insert_neuron_2p(node)
                elif nx.degree(self.som, node) == 3:
                    self._insert_neuron_1p(node)
            else:
                break

    def _insert_neuron_1p(self, node: tuple[int, int]) -> None:
        """Add neuron to the only free position.
        The available positions are:
        - x_i, y_i + 1
        - x_i, y_i - 1
        - x_i + 1, y_i
        - x_i - 1, y_i
        """
        node_x, node_y = node
        nbrs = self.som.adj[node]
        for nbr in [
            (node_x, node_y + 1),
            (node_x, node_y - 1),
            (node_x + 1, node_y),
            (node_x - 1, node_y),
        ]:
            if nbr not in nbrs:
                new_node = nbr
                new_weight = 1.1 * self.som.nodes[node]["weight"]
                self._add_node_to_graph(node=new_node, weight=new_weight)

    def _insert_neuron_2p(self, node: tuple[int, int]) -> None:
        """Add new neuron to the direction with the larger error.

        Case 1:
        o --nb1--nb4
         |   |
        nb2--bo--p1
         |   |
        nb3  p2
        The position P1 is preferable if E(NB4) > (ENB)3,
        otherwise P2 is the choice.

        Case 2:
        o --nb1
         |   |
        nb2--bo--p1
             |
             p2
        When there is no neuron adjacent to P1 and P2 (Fig. 3.b), the
        preferable position is P1 if E(NB1) > E(NB2),otherwise a new
        neuron will be added to P2.


        For the case that the boundary neuron (BO) is not at the corner
        of the grid and there is no neuron adjacent to the available
        positions the preferable position is decided randomly.
        """
        nbr1, nbr2 = self.som.adj[node]
        (nbr1_x, nbr1_y), (nbr2_x, nbr2_y) = nbr1, nbr2
        n_x, n_y = node
        error_nbr1 = self.som.nodes[nbr1]["error"]
        error_nbr2 = self.som.nodes[nbr2]["error"]

        # Case b:
        if error_nbr1 > error_nbr2:
            new_node = (2 * n_x - nbr2_x, 2 * n_y - nbr2_y)
            new_weight = (
                2 * self.som.nodes[node]["weight"] - self.som.nodes[nbr2]["weight"]
            )
        else:
            new_node = (2 * n_x - nbr1_x, 2 * n_y - nbr1_y)
            new_weight = (
                2 * self.som.nodes[node]["weight"] - self.som.nodes[nbr1]["weight"]
            )

        #  Case c: Two opposite neighbors
        if nbr1_x == nbr2_x or nbr1_y == nbr2_y:
            if nbr1_x == nbr2_x:
                new_node = (n_x + 1, n_y)
                new_weight = (
                    2 * self.som.nodes[node]["weight"] - self.som.nodes[nbr2]["weight"]
                )
            else:
                new_node = (n_x, n_y + 1)
                new_weight = (
                    2 * self.som.nodes[node]["weight"] - self.som.nodes[nbr1]["weight"]
                )

        self._add_node_to_graph(node=new_node, weight=new_weight)

    def _insert_neuron_3p(self, bo: tuple[int, int]) -> None:
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

        # new
        bo_x, bo_y = bo
        corner_neighbor_positions = {
            (bo_x + 1, bo_y + 1),
            (bo_x + 1, bo_y - 1),
            (bo_x - 1, bo_y + 1),
            (bo_x - 1, bo_y - 1),
        }

        nb_1 = list(self.som.neighbors(bo))[0]
        # n_nb = list(self.som.neighbors(nb_1))
        corner_neighbors = list(
            corner_neighbor_positions.intersection(set(self.som.neighbors(nb_1)))
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

        self._add_node_to_graph(node=new_node, weight=new_weight)

    def _3p_case_a(
        self,
        nb_1: tuple[int, int],
        bo: tuple[int, int],
        nb_2: tuple[int, int],
        nb_3: tuple[int, int],
    ) -> tuple[tuple[int, int], np.ndarray]:
        if (
            self.som.nodes[nb_1]["error"] > self.som.nodes[nb_2]["error"]
            and self.som.nodes[nb_1]["error"] > self.som.nodes[nb_3]["error"]
        ):
            new_node, new_weight = self._3p_case_c(nb_1, bo)
        elif self.som.nodes[nb_2]["error"] > self.som.nodes[nb_3]["error"]:
            new_node, new_weight = self._3p_case_b(nb_1, bo, nb_2)
        else:
            new_node, new_weight = self._3p_case_b(nb_1, bo, nb_2)

        return new_node, new_weight

    def _3p_case_b(
        self, nb_1: tuple[int, int], bo: tuple[int, int], nb_2: tuple[int, int]
    ) -> tuple[tuple[int, int], np.ndarray]:
        if self.som.nodes[nb_1]["error"] > self.som.nodes[nb_2]["error"]:
            new_node, new_weight = self._3p_case_c(nb_1, bo)
        else:
            new_node = (
                nb_2[0] + bo[0] - nb_1[0],
                nb_2[1] + bo[1] - nb_1[1],
            )

            new_weight = (
                (2 * self.som.nodes[bo]["weight"] - self.som.nodes[nb_1]["weight"])
                + self.som.nodes[nb_2]["weight"]
            ) / 2

        return new_node, new_weight

    def _3p_case_c(
        self, neighbor: tuple[int, int], node: tuple[int, int]
    ) -> tuple[tuple[int, int], np.ndarray]:
        new_node = (2 * node[0] - neighbor[0], 2 * node[1] - neighbor[1])
        new_weight = (
            2 * self.som.nodes[node]["weight"] - self.som.nodes[neighbor]["weight"]
        )
        return new_node, new_weight

    def _add_node_to_graph(self, node: tuple[int, int], weight: np.ndarray) -> None:
        self.som.add_node(node)
        self.som.nodes[node]["weight"] = weight
        self.som.nodes[node]["error"] = 0
        self.som.nodes[node]["epoch_created"] = self.current_epoch
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
            if nbr in self.som.nodes:
                self.som.add_edge(node, nbr)

    def _sigma(self) -> float:
        """Return the neighborhood bandwidth for each epoch.
        If no sigma is given, the starting bandwidth is set to
        0.2 * the square root of the number of neurons in each epoch.
        The ending bandwidth is set to 0.02 * the square root of the
        number of neurons in each epoch.

        Returns:
            float: The neighborhood bandwidth for each epoch.
        """
        epoch = self.current_epoch - 1
        if self.SIGMA_START is None:
            sigma_start = 0.2 * np.sqrt(self.som.number_of_nodes())
        else:
            sigma_start = self.SIGMA_START

        if self.SIGMA_END is None:
            sigma_end = 0.02 * np.sqrt(self.som.number_of_nodes())
        else:
            sigma_end = self.SIGMA_END

        if self.training_phase == "coarse":
            if self.DECAY_FUNCTION == "linear":
                sigma = sigma_start * (
                    1 - (1 / self.COARSE_TRAINING_FRAC * epoch / self.N_EPOCHS)
                ) + sigma_end * (epoch / self.N_EPOCHS)

            elif self.DECAY_FUNCTION == "exponential":
                fac = 1 / self.N_EPOCHS * (log(sigma_end) - log(sigma_start))
                sigma = sigma_start * np.exp(fac * epoch / self.COARSE_TRAINING_FRAC)
        else:
            sigma = 0.1

        return sigma

    def quantization_error(self, data: npt.NDArray[np.float32]) -> float:
        """Return the average distance from each sample to the nearest
        prototype.

        Parameters
        ----------
        data : array_like of shape (n_samples, n_features)
            Data to quantize.

        Returns
        -------
        error : float
            Average distance from each sample to the nearest prototype.
        """
        winners = self._get_winning_neurons(data, n_bmu=1)
        error = np.mean(np.linalg.norm(self.weights[winners] - data, axis=1))
        return error

    def topographic_error(self, data: npt.NDArray[np.float32]) -> float:
        """The topographic error is a measure for the topology preservation of
        the map.

        For each sample we get the two best matching units. If the BMU are
        connected on the grid, there is no error. If the distance is
        larger an error occurred. The total error is the number
        of single errors divided yb the number of samples.

        Parameters
        ----------
        data : array_like of shape (n_samples, n_features)
            Data to show the SOM.

        Returns
        -------
        topographic error : float
            Fraction of samples with topographic errors over all samples.
        """
        bmu_indices = self._get_winning_neurons(data, n_bmu=2).T
        errors = 0
        for bmu_1, bmu_2 in bmu_indices:
            dist = self.distance_matrix[bmu_1, bmu_2]
            if dist > 1:
                errors += 1

        return errors / data.shape[0]

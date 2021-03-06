from math import log
from typing import Any
import numpy as np
import numpy.typing as npt
import networkx as nx
from scipy.spatial.distance import cdist
from tqdm import tqdm


class DBGSOM:
    """A Directed Batch Growing Self-Organizing Map.

    Parameters
    ----------
    sf : float (default = 0.4)
        Spreading factor to calculate the treshold for neuron insertion.

    n_epochs : int (default = 30)
        Number of training epochs.

    sigma_start, sigma_end : {None, numeric}
        Start and end values for the neighborhood bandwidth.
        If None, it is calculated dynamically in each epoch.

    decay_function : {'exponential', 'linear'}
        Decay function to use for sigma.

    coarse_training : float (default = 1)
        Fraction of training epochs to use for coarse training.
        In coarse training, the neighborhood bandwidth is decreased from
        sigma_start to sigma_end. In fine training, the bandwidth is constant at
        sigma_end and no new neurons are added.

    random_state : any (optional, default = None)
        Random state for weight initialization.
    """

    def __init__(
        self,
        n_epochs: int = 30,
        sf: float = 0.4,
        sigma_start: float = None,
        sigma_end: float = None,
        decay_function: str = "exponential",
        coarse_training: float = 1,
        random_state=None,
    ) -> None:
        self.SF = sf
        self.N_EPOCHS = n_epochs
        self.SIGMA_START = sigma_start
        self.SIGMA_END = sigma_end
        self.DECAY_FUNCTION = decay_function
        self.COARSE_TRAINING = coarse_training
        self.RANDOM_STATE = random_state

    def fit(self, data) -> None:
        """Train SOM on training data.

        Parameters
        ----------
        data : array_like, shape = [n_samples, n_features]
            Training data.
        """
        self._initialization(data)
        self._grow(data)

    def _initialization(self, data: npt.NDArray) -> None:
        """First training phase.

        Calculate growing threshold as gt = -data_dimensions * log(spreading_factor).
        Create a graph containing the first four neurons in a square with init vectors.
        """
        data = data.astype(np.float32)
        BATCH_SIZE = np.sqrt(len(data))
        self.N_BATCHES = int(len(data) / BATCH_SIZE)
        data_dimensionality = data.shape[1]
        self.GROWING_TRESHOLD = -data_dimensionality * log(self.SF)
        self.rng = np.random.default_rng(seed=self.RANDOM_STATE)
        self.som = self._create_som(data)
        self.distance_matrix = nx.floyd_warshall_numpy(self.som)
        self.weights = np.array(list(dict(self.som.nodes.data("weight")).values()))
        self.neurons: list[tuple[int, int]] = list(self.som.nodes)

    def _grow(self, data: npt.NDArray) -> None:
        """Second training phase"""
        max_epoch = self.N_EPOCHS
        for i in tqdm(
            iterable=range(self.N_EPOCHS),
            unit=" epochs",
        ):
            self.current_epoch = i + 1
            self.weights = np.array(list(dict(self.som.nodes.data("weight")).values()))
            if len(self.som.nodes) > len(self.neurons) or self.current_epoch == 1:
                self.neurons = list(self.som.nodes)
                self._update_distance_matrix()

            winners = self._get_winning_neurons(data, n_bmu=1)
            self._update_weights(winners, data)
            self._calculate_accumulative_error(winners, data)
            if (
                self.current_epoch != max_epoch
                and self.current_epoch < self.COARSE_TRAINING * max_epoch
            ):
                self._distribute_errors()
                self._add_new_neurons()

    def _create_som(self, data: npt.NDArray) -> nx.Graph:
        """Create a graph containing the first four neurons in a square.
        Each neuron has a weight vector randomly chosen from the training samples.
        """
        init_vectors = self.rng.choice(a=data, size=4, replace=False)
        neurons = [
            ((0, 0), {"weight": init_vectors[0]}),
            ((0, 1), {"weight": init_vectors[1]}),
            ((1, 0), {"weight": init_vectors[2]}),
            ((1, 1), {"weight": init_vectors[3]}),
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

        Return index of winning neuron or best matching units(s) for each sample.
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
        Only paths of length =< 3 * sigma + 1 are considered for performance reasons.
        """
        som = self.som
        sigma = self._sigma()
        n = len(self.neurons)
        m = np.zeros((n, n))
        m.fill(np.inf)
        dist_dict = dict(nx.all_pairs_shortest_path_length(som, cutoff=3 * sigma + 1))
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
            New weight vector = sum(kernel * n_samples * centers) / sum(kernel * n_samples)
        Step 5: Write new weight vectors to the graph.
        """
        voronoi_set_centers = self.weights
        for winner in np.unique(winners):
            voronoi_set_centers[winner] = data[winners == winner].mean(axis=0)

        neuron_counts = np.zeros(shape=len(self.neurons), dtype=np.float32)
        winners, winner_counts = np.unique(winners, return_counts=True)
        for winner, count in zip(winners, winner_counts):
            neuron_counts[winner] = count

        gaussian_kernel = self._gaussian_neighborhood()
        numerator = np.sum(
            voronoi_set_centers
            * neuron_counts[:, np.newaxis]
            * gaussian_kernel[:, :, np.newaxis],
            axis=1,
        )
        denominator = np.sum(
            gaussian_kernel[:, :, np.newaxis] * neuron_counts[:, np.newaxis],
            axis=1,
        )
        new_weights = numerator / denominator

        new_weights_dict = {
            neuron: weight for neuron, weight in zip(self.neurons, new_weights)
        }
        nx.set_node_attributes(G=self.som, values=new_weights_dict, name="weight")

    def _gaussian_neighborhood(self) -> np.ndarray:
        """Calculate the gaussian neighborhood function for all neuron pairs."""
        sigma = self._sigma()
        h = np.exp(-(self.distance_matrix**2 / (2 * sigma**2))).astype(np.float32)

        return h

    def _calculate_accumulative_error(
        self, winners: np.ndarray, data: npt.NDArray
    ) -> None:
        """Get the quantization error for each neuron
        and save it as "error" to the graph.
        """
        for winner in range(len(self.neurons)):
            samples = data[winners == winner]
            dist = np.linalg.norm(self.weights[winner] - samples, axis=1)
            error = dist.sum()
            self.som.nodes[self.neurons[winner]]["error"] = error

    def _distribute_errors(self) -> None:
        """For each neuron i which is not a boundary neuron and E_i > GT,
        a half value of E_i is equally distributed to the neighboring
        boundary neurons if exist.
        """
        for node, neighbors in self.som.adj.items():
            if len(neighbors.items()) == 4:
                is_boundary = False
            else:
                is_boundary = True

            if not is_boundary:
                node_error = self.som.nodes[node]["error"]
                n_boundary_neighbors = 0
                for neighbor in neighbors.keys():
                    if len(self.som.adj[neighbor].items()) < 4:
                        n_boundary_neighbors += 1

                for neighbor in neighbors.keys():
                    if len(self.som.adj[neighbor].items()) < 4:
                        self.som.nodes[neighbor]["error"] += (
                            0.5 * node_error / n_boundary_neighbors
                        )

    def _add_new_neurons(self) -> None:
        """Add new neurons to places where the error is above
        the growing threshold.
        """
        for node in self.neurons:
            if self.som.nodes[node]["error"] > self.GROWING_TRESHOLD:
                if nx.degree(self.som, node) == 1:
                    self._insert_neuron_3p(node)
                elif nx.degree(self.som, node) == 2:
                    self._insert_neuron_2p(node)
                elif nx.degree(self.som, node) == 3:
                    self._insert_neuron_1p(node)

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
                self.som.add_node(nbr)
                self.som.nodes[nbr]["weight"] = 1.1 * self.som.nodes[node]["weight"]
                self.som.nodes[nbr]["error"] = 0
                self.som.nodes[nbr]["epoch_created"] = self.current_epoch
                self._add_new_connections(nbr)

    def _insert_neuron_2p(self, node: tuple[int, int]) -> None:
        """Add new neuron to the side with greater error."""
        nbr1, nbr2 = self.som.adj[node]
        (nbr1_x, nbr1_y), (nbr2_x, nbr2_y) = nbr1, nbr2
        n_x, n_y = node
        #  Case c: Two opposite neighbors
        if nbr1_x == nbr2_x or nbr1_y == nbr2_y:
            if nbr1_x == nbr2_x:
                new_node = (n_x, n_y + 1)
                new_weight = (
                    2 * self.som.nodes[node]["weight"] - self.som.nodes[nbr2]["weight"]
                )
            else:
                new_node = (n_x + 1, n_y)
                new_weight = (
                    2 * self.som.nodes[node]["weight"] - self.som.nodes[nbr1]["weight"]
                )
        #  Case b: Two neurons with no adjacent neurons
        else:
            nbr1_err = self.som.nodes[nbr1]["error"]
            nbr2_err = self.som.nodes[nbr2]["error"]
            if nbr1_err > nbr2_err:
                new_node = (2 * n_x - nbr2_x, 2 * n_y - nbr2_y)
                new_weight = (
                    2 * self.som.nodes[node]["weight"] - self.som.nodes[nbr2]["weight"]
                )
            else:
                new_node = (2 * n_x - nbr1_x, 2 * n_y - nbr1_y)
                new_weight = (
                    2 * self.som.nodes[node]["weight"] - self.som.nodes[nbr1]["weight"]
                )

        self.som.add_node(new_node)
        self.som.nodes[new_node]["weight"] = new_weight
        self.som.nodes[new_node]["error"] = 0
        self.som.nodes[new_node]["epoch_created"] = self.current_epoch
        self._add_new_connections(new_node)

    def _insert_neuron_3p(self, node: tuple[int, int]) -> None:
        """When the neuron has three free available positions, add the new neuron to the opposite
        side of the existing neighbor.
        """
        neighbor = list(self.som.neighbors(node))[0]
        new_node = (2 * node[0] - neighbor[0], 2 * node[1] - neighbor[1])
        new_weight = (
            2 * self.som.nodes[node]["weight"] - self.som.nodes[neighbor]["weight"]
        )

        self.som.add_node(new_node)
        self.som.nodes[new_node]["weight"] = new_weight
        self.som.nodes[new_node]["error"] = 0
        self.som.nodes[new_node]["epoch_created"] = self.current_epoch
        self._add_new_connections(new_node)

    def _add_new_connections(self, node: tuple[int, int]) -> None:
        """Given a node (x, y), add new connections to the neighbors of the node, if exist."""
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
        0.2 * the squareroot of the number of neurons in each epoch.
        The ending bandwidth is set to 0.05 * the squareroot of the
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
            sigma_end = 0.05 * np.sqrt(self.som.number_of_nodes())
        else:
            sigma_end = self.SIGMA_END

        if epoch < self.N_EPOCHS * self.COARSE_TRAINING:
            if self.DECAY_FUNCTION == "linear":
                sigma = sigma_start * (
                    1 - (1 / self.COARSE_TRAINING * epoch / self.N_EPOCHS)
                ) + sigma_end * (epoch / self.N_EPOCHS)

            elif self.DECAY_FUNCTION == "exponential":
                fac = 1 / self.N_EPOCHS * (log(sigma_end) - log(sigma_start))
                sigma = sigma_start * np.exp(fac * 1 / self.COARSE_TRAINING * epoch)
        else:
            sigma = sigma_end

        return sigma

    def quantization_error(self, data: npt.NDArray[np.float32]) -> float:
        """Return the average distance from each sample to the nearest prototype.

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
        """The topographic error is a measure for the topology preservation of the map.

        For each sample we get the two best matching units. If the BMU are connected on the grid,
        there is no error. If the distance is larger an error occurred. The total error is the number
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
        sample_bmus = self._get_winning_neurons(data, n_bmu=2)
        errors = 0
        for sample in sample_bmus.T:
            dist = self.distance_matrix[sample[0], sample[1]]
            if dist > 1:
                errors += 1

        return errors / data.shape[0]

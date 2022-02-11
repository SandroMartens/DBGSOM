from math import log
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist

class DBGSOM:
    """A Directed Batch Growing Self-Organizing Map.

    Parameters
    ----------
    sf : float (default = 0.4)
        Spreading factor to calculate the treshold for neuron insertion.

    n_epochs : int (default = 30)
        Number of training epochs.

    sigma : float (optional)
        Neighborhood bandwidth.

    random_state: any (optional)
        Random state for weight initialization.
    """
    def __init__(
        self,
        n_epochs: int = 30,
        sf: float = 0.4,
        sigma: float = 1,
        random_state = None
    ) -> None:
        self.SF = sf
        self.N_EPOCHS = n_epochs
        self.SIGMA = sigma
        self.RANDOM_STATE = random_state

    def train(self, data):
        """Train SOM on training data."""
        self.initialization(data)
        self.grow(data)

    def initialization(self, data) -> None:
        """First training phase.

        Calculate growing threshold as gt = -data_dimensions * log(spreading_factor).
        Create a graph containing the first four neurons in a square with init vectors.
        """
        data = data.astype(np.float32)
        data_dimensionality = data.shape[1]
        self.GROWING_TRESHOLD = -data_dimensionality * log(self.SF)
        self.rng = np.random.default_rng(seed=self.RANDOM_STATE)
        self.som = self.create_som(data)
        self.distance_matrix = nx.floyd_warshall_numpy(self.som)
        #  Get array with neurons as index and values as columns
        self.weights = np.array(
            list(dict(self.som.nodes.data("weight")).values())
            )
        #  List of node indices
        self.neurons = list(self.som.nodes)

    def grow(self, data):
        """Second training phase"""
        max_epoch = self.N_EPOCHS
        for i in range(self.N_EPOCHS):
            self.current_epoch = i + 1
            #  Get array with neurons as index and values as columns
            self.weights = np.array(
                list(dict(self.som.nodes.data("weight")).values())
            )
            #  List of node indices
            if (len(self.som.nodes) > len(self.neurons) 
                or self.current_epoch == 1):
                self.neurons = list(self.som.nodes)
                self.update_distance_matrix()

            winners = self.get_winning_neurons(data, n_bmu=1)
            self.weights = self.update_weights(winners, data)
            self.calculate_accumulative_error(winners, data)
            #  Only add new neurons in each 5th epoch. This leads to a lower topographic error.
            if (
                self.current_epoch < 0.5 * max_epoch and
                self.current_epoch % 3 == 0
            ):
                self.distribute_errors()
                self.add_new_neurons(data)

    def create_som(self, data) -> nx.Graph:
        """Create a graph containing the first four neurons in a square. 
        Each neuron has a weight vector randomly chosen from the training samples.
        """
        init_vectors = self.rng.choice(a=data, size=4, replace=False)
        neurons = [
            ((0, 0), {"weight": init_vectors[0]}),
            ((0, 1), {"weight": init_vectors[1]}),
            ((1, 0), {"weight": init_vectors[2]}),
            ((1, 1), {"weight": init_vectors[3]})
        ]

        #  Build a square
        edges = [
            ((0, 0), (0, 1)),
            ((0, 0), (1, 0)),
            ((1, 0), (1, 1)),
            ((0, 1), (1, 1))
        ]

        som = nx.Graph()
        som.add_nodes_from(neurons)
        som.add_edges_from(edges)

        return som

    def get_winning_neurons(self, data, n_bmu:int) -> np.ndarray:
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

    def update_distance_matrix(self):
        """Update distance matrix between neurons. 
        Only paths of length =< 3 * sigma are considered for performance reasons.
        """
        som = self.som
        sigma = self.reduce_sigma()
        n = len(self.neurons)
        m = np.zeros((n, n))
        m.fill(np.inf)
        dist_dict = dict(nx.all_pairs_shortest_path_length(som, cutoff=2*sigma + 1))
        for i1, neuron1 in enumerate(self.neurons):
            for i2, neuron2 in enumerate(self.neurons):
                if neuron1 in dist_dict.keys():
                    if neuron2 in dist_dict[neuron1].keys():
                        m[i1, i2] = dist_dict[neuron1][neuron2]

        self.distance_matrix = m

    def update_weights(self, winners, data) -> np.ndarray:
        """Update the weight vectors according to the batch learning rule.

        Step 1: Calculate the center of the voronoi set of the neurons.
        Step 2: Count the number of samples in each voronoi set.
        Step 3: Calculate the kernel function for all neuron pairs.
        Step 4: New weight vector = sum(kernel * n_samples * centers) / sum(kernel * n_samples)
        """
        voronoi_set_centers = self.weights
        for winner in np.unique(winners):
            voronoi_set_centers[winner] = data[winners == winner].mean(axis=0)

        neuron_counts = np.ones(shape=len(self.neurons), dtype=np.float32)
        winners, winner_counts = np.unique(winners, return_counts=True)
        for winner, count in zip(winners, winner_counts):
            neuron_counts[winner] = count

        gaussian_kernel = self.gaussian_neighborhood()
        numerator = np.sum(
            voronoi_set_centers * 
            neuron_counts[:, np.newaxis] * 
            gaussian_kernel[:,:,np.newaxis],
            axis=1
        )
        denominator = np.sum(
            gaussian_kernel[:,:,np.newaxis] * 
            neuron_counts[:,np.newaxis],
            axis=1
        )
        new_weights = (numerator / denominator)

        new_weights_dict = {neuron: weight for neuron, weight in zip(self.neurons, new_weights)}
        nx.set_node_attributes(
            G=self.som,
            values=new_weights_dict,
            name="weight")

        return new_weights

    def gaussian_neighborhood(self) -> np.ndarray:
        """Calculate the gaussian neighborhood function for all neuron pairs."""
        sigma = self.reduce_sigma()
        h = np.exp(-(self.distance_matrix**2 / (2*sigma**2))).astype(np.float32)
        # if self.current_epoch % 5 == 0:
        #     h = np.exp(-(self.distance_matrix**2 / (2*sigma**2)))
        # else:
        #     h = np.identity(self.distance_matrix.shape[0])

        return h

    def calculate_accumulative_error(self, winners, data) -> None:
        """Get the quantization error for each neuron 
        and save it as "error" to the graph.
        """
        for winner in range(len(self.neurons)):
            samples = data[winners == winner]
            dist = np.linalg.norm(self.weights[winner] - samples, axis=1)
            error = dist.sum()
            self.som.nodes[self.neurons[winner]]["error"] = error

    def distribute_errors(self):
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
                            0.5*node_error / n_boundary_neighbors
                        )

    def add_new_neurons(self, data) -> None:
        """Add new neurons to places where the error is above 
        the growing threshold.
        """
        for node in self.neurons:
            if nx.degree(self.som, node) == 1:
                if self.som.nodes[node]["error"] > self.GROWING_TRESHOLD:
                    self.insert_neuron_3p(node)
            elif nx.degree(self.som, node) == 2:
                if self.som.nodes[node]["error"] > self.GROWING_TRESHOLD:
                    self.insert_neuron_2p(node, data)
            elif nx.degree(self.som, node) == 3:
                if self.som.nodes[node]["error"] > self.GROWING_TRESHOLD:
                    self.insert_neuron_1p(node, data)

    def insert_neuron_1p(self, node: tuple, data) -> None:
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
            (node_x, node_y+1),
            (node_x, node_y-1),
            (node_x+1, node_y),
            (node_x-1, node_y)]:
            if nbr not in nbrs:
                self.som.add_node(nbr)
                self.som.nodes[nbr]["weight"] = 1.1 * self.som.nodes[node]["weight"] 
                self.som.nodes[nbr]["error"] = 0
                self.add_new_connections(nbr)

    def insert_neuron_2p(self, node: tuple, data) -> None:
        """Add new neuron to the side with greater error."""
        nbr1, nbr2 = self.som.adj[node]
        (nbr1_x, nbr1_y), (nbr2_x, nbr2_y) = nbr1, nbr2
        n_x, n_y = node
        #  Case c: Two opposite neighbors
        if (nbr1_x == nbr2_x or nbr1_y == nbr2_y):
            if nbr1_x == nbr2_x:
                new_node = (n_x, n_y+1)
                new_weight = 2 * self.som.nodes[node]["weight"] - self.som.nodes[nbr2]["weight"]
            else:
                new_node = (n_x+1, n_y)
                new_weight = 2 * self.som.nodes[node]["weight"] - self.som.nodes[nbr1]["weight"]
        #  Case b: Two neurons with no adjacent neurons
        else:
            nbr1_err = self.som.nodes[(nbr1_x, nbr1_y)]["error"]
            nbr2_err = self.som.nodes[(nbr2_x, nbr2_y)]["error"]
            if nbr1_err > nbr2_err:
                new_node = (n_x + (n_x-nbr2_x), n_y + (n_y-nbr2_y))
                new_weight = 2 * self.som.nodes[node]["weight"] - self.som.nodes[nbr2]["weight"]
            else:
                new_node = (n_x + (n_x-nbr1_x), n_y + (n_y-nbr1_y))
                new_weight = 2 * self.som.nodes[node]["weight"] - self.som.nodes[nbr1]["weight"]

        self.som.add_node(new_node)
        self.som.nodes[new_node]["weight"] = new_weight
        self.som.nodes[new_node]["error"] = 0
        self.add_new_connections(new_node)

    def insert_neuron_3p(self, node: tuple) -> None:
        """When the neuron has three free available positions, add the new neuron to the opposite
        side of the existing neighbor.
        """
        neighbor = list(self.som.neighbors(node))[0]
        new_node = (2 * node[0] - neighbor[0], 2 * node[1] - neighbor[1])
        new_weight = 2 * self.som.nodes[node]["weight"] - self.som.nodes[neighbor]["weight"]

        self.som.add_node(new_node)
        self.som.nodes[new_node]["weight"] = new_weight
        self.som.nodes[new_node]["error"] = 0
        self.add_new_connections(new_node)

    def add_new_connections(self, node: tuple) -> None:
        """Add edges from new neuron to existing neighbors."""
        node_x, node_y = node
        for nbr in [
            (node_x, node_y+1),
            (node_x, node_y-1),
            (node_x-1, node_y),
            (node_x+1, node_y)
        ]:
            if nbr in self.som.nodes:
                self.som.add_edge(node, nbr)

    def reduce_sigma(self) -> float:
        """Return the neighborhood bandwidth for each epoch.
        If no sigma is given, the starting bandwidth is set to 
        0.5 * the squareroot of the number of neurons in each epoch.
        The ending bandwidth is set to 0.5.

        We have two phases: In the first half, we have a decreasing sigma from sigma_max to 0.5.
        In the second half, sigma stays constant at 0.5.

        Returns:
            float: The neighborhood bandwidth for each epoch.
        """
        epoch = self.current_epoch
        if self.SIGMA is None:
            sigma_zero = 0.5 * np.sqrt(self.som.number_of_nodes())
        else:
            sigma_zero = self.SIGMA

        sigma = sigma_zero * (1 - epoch / self.N_EPOCHS) +0.6 * (epoch / self.N_EPOCHS)
        if epoch < 0.5 * self.N_EPOCHS:
            sigma = sigma_zero * (1-(2*epoch/self.N_EPOCHS)) + 0.6 * (2*epoch/self.N_EPOCHS)
        else:
            sigma = 0.6
        # print(sigma)
        # print(0.5 * np.sqrt(self.som.number_of_nodes()))

        return sigma

    def quantization_error(self, data) -> float:
        """Get the average distance from each sample to the nearest prototype.

        Parameters
        ----------
        data : ndarray
            data to cluster

        Returns
        -------
        error: float
            average distance from each sample to the nearest prototype
        """
        winners = self.get_winning_neurons(data, n_bmu=1)
        error = 0
        for sample, winner in zip(data, winners):
            error += np.linalg.norm(self.weights[winner] - sample)

        return error/len(data)

    def topographic_error(self, data) -> float:
        """The topographic error is a measure for the topology preservation of the map.
        
        For each sample we get the two best matching units. If the BMU are connected on the grid,
        there is no error. If the distance is larger an error occurred. The total error is the number
        of single errors divided yb the number of samples.

        Parameters
        ----------
        data : ndarray
            Data to show the SOM.

        Returns
        -------
        topographic error: float
            Fraction of samples with topographic errors over all samples.
        """
        sample_bmus = self.get_winning_neurons(data, n_bmu=2)
        errors = 0
        for sample in sample_bmus.T:
            dist = self.distance_matrix[sample[0], sample[1]]
            if dist > 1:
                errors += 1

        return errors/data.shape[0]

from math import log
import numpy as np
import networkx as nx
import pandas as pd
from random import random


class DBGSOM:
    """A Directed Batch Growing Self-Organizing Map.

    Parameters
    ----------
    sf : float (default = 0.8)
        Spreading factor to calculate the treshold for neuron insertion.

    n_epochs : int
        Number of training epochs.

    sigma : float (Default = 1)
        Neighborhood bandwidth.

    random_state: any
        Random state for weight initialization.
    """
    def __init__(
        self,
        n_epochs: int,
        sf: float = 0.8,
        sigma: float = 1,
        random_state = None
    ) -> None:
        self.SF = sf
        self.n_epochs = n_epochs
        self.sigma = sigma
        self.random_state = random_state

    def train(self, data):
        """Train SOM on training data."""
        self.initialization(data)
        self.grow(data)

    def initialization(self, data):
        """First training phase.

        Initialize neurons in a square topology with random weights.
        Calculate growing threshold.
        Specify number of training epochs.
        """
        D = data.shape[1]
        self.growing_treshold = -D * log(self.SF)
        rng = np.random.default_rng(seed=self.random_state)
        #  Use four random points as initialization
        init_vectors = rng.choice(a=data, size=4)
        self.som = self.create_som(init_vectors)
        #  Get array with neurons as index and values as columns
        self.weights = np.array(
            list(dict(self.som.nodes.data("weight")).values())
            )
        #  List of node indices
        self.neurons = list(self.som.nodes)

    def grow(self, data):
        """Second training phase"""
        for i in range(self.n_epochs):
            self.current_epoch = i
            #  Get array with neurons as index and values as columns
            self.weights = np.array(
                list(dict(self.som.nodes.data("weight")).values())
            )
            #  List of node indices
            self.neurons = list(self.som.nodes)
            winners = self.get_winning_neurons(data, n_bmu=1)
            self.pt_distances = self.prototype_distances()
            self.weights = self.update_weights(winners, data)
            self.calculate_accumulative_error(winners, data)
            self.distribute_errors()
            self.add_new_neurons()
            self.allocate_new_weights()

        #  Save the final weights
        self.weights = self.update_weights(winners, data)
        self.weights = np.array(
            list(dict(self.som.nodes.data("weight")).values())
            )
        self.neurons = list(self.som.nodes)
        self.pt_distances = self.prototype_distances()

    def create_som(self, init_vectors: np.ndarray) -> nx.Graph:
        """Create a graph containing the first four neurons."""
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
        distances = np.linalg.norm(self.weights[:, np.newaxis, :] - data, axis=2)
        if n_bmu == 1:
            winners = np.argmin(distances, axis=0)
        else: 
            winners = np.argsort(distances, axis=0)[:n_bmu]

        return winners

    def prototype_distances(self) -> pd.DataFrame:
        """Return distance (shortest path) of
        two prototypes on the som.
        """
        g = self.som
        nodes = list(g)
        return pd.DataFrame(
            nx.floyd_warshall_numpy(g),
            index=nodes, 
            columns=nodes
        )

    def update_weights(self, winners, data) -> np.ndarray:
        """The updated weight vectors of the neurons
        are calculated by the batch learning principle.
        """
        voronoi_set_centers = np.empty_like(self.neurons, dtype="float32")
        for winner in np.unique(winners):
            voronoi_set_centers[winner] = data[winners == winner].mean(axis=0)

        neuron_counts = np.zeros(shape=len(self.neurons))
        winners, winner_counts = np.unique(winners, return_counts=True)
        for winner, count in zip(winners, winner_counts):
            neuron_counts[winner] = count

        gaussian_kernel = self.gaussian_neighborhood()
        new_weights = np.empty_like(self.weights)
        for i in range(len(new_weights)):
            numerator = gaussian_kernel.iloc[i].to_numpy() * neuron_counts * voronoi_set_centers.T
            denumerator = (gaussian_kernel.iloc[i].to_numpy() * neuron_counts).sum()
            new_weights[i] = (numerator.sum(axis=1) / denumerator)

        new_weights_dict = {neuron: weight for neuron, weight in zip(self.neurons, new_weights)}
        nx.set_node_attributes(
            G=self.som,
            values=new_weights_dict,
            name="weight")

        return new_weights

    def gaussian_neighborhood(self) -> pd.DataFrame:
        """Return gaussian kernel of two prototypes."""
        sigma = self.reduce_sigma()
        h = np.exp(-(self.pt_distances**2 / (2*sigma**2)))
        return h

    def calculate_accumulative_error(self, winners, data) -> None:
        """Get the quantization error for each neuron 
        and save it to the graph.
        """
        for winner in range(len(self.neurons)):
            samples = data[winners == winner]
            dist = np.linalg.norm(self.weights[winner] - samples, axis=1)
            error = dist.sum()
            # errors[winner] = error
            self.som.nodes[self.neurons[winner]]["error"] = error

    def distribute_errors(self):
        pass

    def add_new_neurons(self) -> None:
        """Add new neurons to places where the error is above 
        the growing threshold.
        """
        for node in self.neurons:
            if nx.degree(self.som, node) == 1:
                if self.som.nodes[node]["error"] > self.growing_treshold:
                    self.insert_neuron_1p(node)
                    # self.set_weight_1p(node)
            elif nx.degree(self.som, node) == 2:
                if self.som.nodes[node]["error"] > self.growing_treshold:
                    self.insert_neuron_2p(node)
            elif nx.degree(self.som, node) == 3:
                if self.som.nodes[node]["error"] > self.growing_treshold:
                    self.insert_neuron_3p(node)

    def insert_neuron_1p(self, node: tuple) -> None:
        """If only one position is free, add new neuron to that position."""
        node_x, node_y = node
        nbrs = self.som.adj[node]
        for nbr in [
                    (node_x, node_y+1),
                    (node_x, node_y-1),
                    (node_x-1, node_y),
                    (node_x+1, node_y)]:
            if nbr not in nbrs:
                self.som.add_node(nbr)
                self.som.nodes[nbr]["weight"] = 2*self.som.nodes[node]["weight"] + 1
                self.som.nodes[nbr]["error"] = 0
                self.add_new_connections(nbr)

    def insert_neuron_2p(self, node: tuple) -> None:
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
                new_weight = 2 * self.som.nodes[node]["weight"] - self.som.nodes[nbr2]["weight"]
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

    def insert_neuron_3p(self, node: tuple):
        pass
        # self.insert_neuron_2p(node)

    def add_new_connections(self, node: tuple) -> None:
        """Add edges from new neuron to existing neighbors."""
        node_x, node_y = node
        for nbr in [
                    (node_x, node_y+1),
                    (node_x, node_y-1),
                    (node_x-1, node_y),
                    (node_x+1, node_y)]:
            if nbr in self.neurons:
                self.som.add_edge(node, nbr)

    def allocate_new_weights(self):
        pass

    def reduce_sigma(self) -> float:
        """Decay bandwidth in each epoch"""
        epoch = self.current_epoch

        if epoch%2 != 0:
            sigma = 0.1
            # sigma = self.sigma * np.exp(-epoch/self.n_epochs)
        else:
            sigma = self.sigma * np.exp(-epoch/self.n_epochs)

        return sigma

    def quantization_error(self, data):
        """Get the average distance from each sample to the nearest prototype.
        
        Parameters
        ----------
        
        data : ndarray
            data to cluster
            """
        winners = self.get_winning_neurons(data, n_bmu=1)
        error_sum = 0
        for sample, winner in zip(data, winners):
            error = np.linalg.norm(self.weights[winner] - sample)
            error_sum += error

        return error_sum/len(data)

    def topographic_error(self, data):
        """The topographic error is a measure for the topology preservation of the map.
        
        For each sample we get the two best matching units. If the BMU are connected on the grid,
        there is no error. If the distance is larger an error occured. The total error is the number
        of single errors divided yb the number of samples.

        Parameters
        ----------
        data : ndarray
            Data to show the SOM.
        """
        sample_bmus = self.get_winning_neurons(data, n_bmu=2)
        errors = 0
        for sample in sample_bmus.T:
            x = self.neurons[sample[0]]
            y = self.neurons[sample[1]]
            dist = self.pt_distances[x][y]
            if dist > 1:
                errors += 1

        # for sample in sample_bmus.T:
        #     dist = self.pt_distances.iloc[sample[0], sample[1]]
        #     if dist > 1:
        #         errors += 1

        return errors/data.shape[0]

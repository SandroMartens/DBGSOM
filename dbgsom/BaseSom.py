"""Handles the core SOM functionality."""

import copy
import sys
from math import exp, log, sqrt
from numbers import Integral
from typing import Any, Self

from sklearn.utils.validation import validate_data

# from matplotlib import pyplot as plt
# import matplotlib

try:
    import matplotlib.pyplot as plt
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
    from sklearn.utils import check_random_state
    from sklearn.utils._param_validation import Interval, StrOptions
    from sklearn.utils.multiclass import check_classification_targets
    from sklearn.utils.validation import check_is_fitted
    from tqdm import tqdm
except ImportError as e:
    print(e)
    sys.exit()


class BaseSom(BaseEstimator):
    """Base Self-Organizing Map (SOM) implementation.

    A Self-Organizing Map is an unsupervised neural network model that maps
    high-dimensional input data onto a low-dimensional, typically 2D or 3D
    grid of neurons. The neurons are updated during training to preserve the
    topological properties of the input space.

    Parameters
    ----------
    n_iter : int, default=200
        Maximum number of iterations for training.

    convergence_iter : int, default=1
        Number of iterations to check for convergence.

    spreading_factor : float, default=0.5
        Factor controlling the spread of neuron activation.

    sigma_start : float or None, default=None
        Initial standard deviation for the neighborhood function.

    sigma_end : float or None, default=None
        Final standard deviation for the neighborhood function.

    vertical_growth : bool, default=False
        Whether to allow vertical growth of the map.

    decay_function : str, default="exponential"
        Decay function for the learning rate and neighborhood. Options: "exponential", "linear".

    learning_rate : float, default=0.02
        Learning rate for weight updates.

    verbose : bool, default=False
        Whether to print training progress.

    coarse_training_frac : float, default=0.5
        Fraction of training data for coarse training phase.

    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility.

    convergence_treshold : float, default=1e-5
        Threshold for convergence criterion.

    max_neurons : int, default=100
        Maximum number of neurons allowed in the map.

    metric : str, default="euclidean"
        Distance metric used for computations.

    threshold_method : str, default="se"
        Method for threshold calculation.

    growth_criterion : str, default="quantization_error"
        Criterion for neuron growth decision.

    min_samples_vertical_growth : int, default=100
        Minimum number of samples required for vertical growth.

    n_jobs : int, default=1
        Number of parallel jobs for computation.

    """

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
        verbose: bool = False,
        coarse_training_frac: float = 0.5,
        random_state: int | None | np.random.RandomState = None,
        convergence_treshold: float = 10**-5,
        max_neurons: int = 100,
        metric: str = "euclidean",
        threshold_method: str = "se",
        growth_criterion: str = "quantization_error",
        min_samples_vertical_growth: int = 100,
        n_jobs: int = 1,
    ) -> None:
        super().__init__()
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
        "n_iter": [Interval(Integral, 1, None, closed="left")],  # type: ignore
        "max_neurons": [Interval(Integral, 4, None, closed="left")],  # type: ignore
        "decay_function": [StrOptions({"exponential", "linear"})],  # type: ignore
    }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
        # tags.transformer_tags = True

    def fit(self, X: npt.ArrayLike, y: None | npt.ArrayLike = None) -> Self:
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

        if y is None:
            X = validate_data(self, X, ensure_min_samples=4)

        elif y is not None:
            X, y = validate_data(self, X, y, ensure_min_samples=4)
            check_classification_targets(y)
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
        self._write_edge_statistics()
        self._delete_dead_neurons_from_graph(X)
        self._label_prototypes(X, y)

        # Vertical growing phase
        if self.vertical_growth:
            self._grow_vertical(X, y)

        self._fit(X)
        # self.labels_ = self.predict(X)
        self.n_iter_ = self._current_epoch

        return self

    def _fit(self, X):
        # For VQ Subclass specific code
        pass

    def predict(self, X):
        raise NotImplementedError

    def _grow_vertical(self, X: npt.NDArray, y: None | npt.NDArray = None) -> None:
        """Triggers vertical growth in the SOM by creating new instances of the DBGSOM
        class and fitting them with filtered data.
        """
        # todo: refactor in sub classes
        self.vertical_growing_threshold_ = 1.5 * self.growing_threshold_
        _, winners = self._get_winning_neurons(X, n_bmu=1)
        relevant_nodes = [
            node
            for (node, error) in self.som_.nodes(data="error")
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
        self, X: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Write the following statistics as attributes to the graph:

        1. Local density: Use a Gaussian kernel to estimate the local density around
        each prototype. Use the average distance from all prototype to their neighbors
        as bandwith sigma.

        2. Hit count: How many samples each prototype represents.

        3. average distance: average distance from each prototype to their neighbors.
        used for plotting the u matrix
        """
        distances, winners = self._get_winning_neurons(X, n_bmu=1)
        average_distances = self._get_u_matrix()

        n_neurons = len(self.neurons_)
        sigma = average_distances.mean()

        hit_counts = np.bincount(winners, minlength=n_neurons)
        kernel_values = np.exp(-(distances**2) / (2 * sigma**2)) / (
            sigma * np.sqrt(2 * np.pi)
        )
        sum_densities = np.bincount(winners, weights=kernel_values, minlength=n_neurons)

        densities = np.divide(
            sum_densities, hit_counts, out=np.zeros(n_neurons), where=hit_counts > 0
        )

        return average_distances, densities, hit_counts

    def _write_node_statistics(self, X: npt.NDArray) -> None:
        average_distances, densities, hit_counts = self._calculate_node_statistics(X)

        for density, hit_count, average_distance, node in zip(
            densities, hit_counts, average_distances, self.som_.nodes
        ):
            self.som_.nodes[node]["density"] = density
            self.som_.nodes[node]["hit_count"] = hit_count
            self.som_.nodes[node]["average_distance"] = average_distance

    def _write_edge_statistics(self) -> None:
        som = self.som_

        for u, v in som.edges:
            # Gewichte der beiden verbundenen Knoten holen
            weight_x = som.nodes[u]["weight"]
            weight_y = som.nodes[v]["weight"]

            # Euklidischen Abstand berechnen
            distance = np.linalg.norm(weight_x - weight_y)

            # Abstand als neues Kanten-Attribut (z.B. "weight_distance") speichern
            som.edges[u, v]["weight_distance"] = 1 / float(distance)

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

    def transform(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> np.ndarray:
        """Calculate a non negative least squares mixture model of prototypes that approximate each sample.

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

        """  # noqa: E501
        check_is_fitted(self)
        if y is None:
            X = validate_data(self, X, reset=False)
        elif y is not None:
            X, y = validate_data(self, X, y, reset=False)
        transformer = SparseCoder(
            dictionary=normalize(self.weights_),
            n_jobs=self.n_jobs,
            positive_code=True,
            transform_alpha=0,
            transform_algorithm="lasso_lars",
        )
        coefs = transformer.transform(normalize(X))
        return coefs

    def plot(
        self,
        color: None | str = None,
        palette: str = "magma_r",
        pointsize: None | str = None,
    ) -> None:
        """Plot the neurons.

        Parameters
        ----------
        color : {'label', 'epoch_created', 'error', 'average_distance', 'density', 'hit_count'}, optional
            Attribute which is represented as color.

        pointsize : {'label', 'epoch_created', 'error', 'average_distance', 'density', 'hit_count'}, optional
            Determines the property used for the node sizing:

            * 'label': Label of the prototype when trained supervised.
            * 'epoch_created': When the neuron was created.
            * 'error' : Quantization error of each neuron.
            * 'average_distance' : Average distance to neighbor neurons in the input space.
            * 'density' : Estimated local density around the prototype.
            * 'hit_count' : How many samples the prototype represents.

        palette : str, default: 'magma_r'
            Name of seaborn palette to color code the values of attribute.

        """
        data = pd.DataFrame(dict(self.som_.nodes)).T.set_index(
            np.arange(len(self.som_.nodes))
        )

        data["label_index"] = pd.to_numeric(data["label"])
        # data["label"] = self.classes_[data["label_index"]]

        data["epoch_created"] = pd.to_numeric(data["epoch_created"])
        data["error"] = pd.to_numeric(data["error"])
        data["density"] = pd.to_numeric(data["density"])
        data["hit_count"] = pd.to_numeric(data["hit_count"])
        data["average_distance"] = pd.to_numeric(data["average_distance"])
        # coordinates = pd.DataFrame(np.array(self.neurons_), columns=["x", "y"])
        coordinates = pd.DataFrame(np.array(self.neurons_), columns=["x", "y"])
        dots = pd.concat([coordinates, data], axis=1)

        # Mappings (Spalten) und Literale (feste Werte) sauber trennen
        plot_kwargs = {}
        dot_kwargs = {}

        # 1. Farbe absichern
        if color is not None and color in dots.columns:
            # Falls eine numerische Spalte überall denselben Wert hat (z.B. alle Fehler = 0),
            # als String formatieren, um die kontinuierliche Division durch Null zu verhindern.
            if (
                pd.api.types.is_numeric_dtype(dots[color])
                and dots[color].nunique() <= 1
            ):
                dots[color] = dots[color].astype(str)
            plot_kwargs["color"] = color

        # 2. Punktgröße absichern
        if isinstance(pointsize, str) and pointsize in dots.columns:
            if (
                pd.api.types.is_numeric_dtype(dots[pointsize])
                and dots[pointsize].nunique() <= 1
            ):
                dots[pointsize] = dots[pointsize].astype(str)
            plot_kwargs["pointsize"] = pointsize

        # Plot sicher aufbauen
        p = so.Plot(dots, x="x", y="y", **plot_kwargs).add(so.Dot(**dot_kwargs))

        # Skalierung nur anwenden, wenn auch nach Farbe gruppiert wird
        if "color" in plot_kwargs:
            p = p.scale(color=palette)

        p.show()

        # p.label(x="", y="").show()

    #     dots = pd.concat([coordinates, data], axis=1)
    #     # matplotlib.use("nbAgg")
    #     # f = plt.figure()
    #     so.Plot(dots, x="x", y="y", color=color, pointsize=pointsize).add(
    #         so.Dot()
    #     ).scale(color=palette).label(x="", y="").show()  # .on(f)
    #     # f
    #     # f.show()
    #     # plt.show()

    def plot_graph(
        self,
        color: str | None = None,
        size: str | None = None,
        layout: str = "spring_weighted",
        seed: int | None = 0,
        min_size: float = 50,
        scale_factor: float = 5,
        palette: str = "tab10",
        ax: "plt.Axes | None" = None,
        figsize: tuple[int, int] = (10, 8),
    ) -> "plt.Axes":
        """Plot the SOM topology as a NetworkX graph with Matplotlib.

        Nodes are placed according to *layout*.  Edges reflect the SOM
        neighbourhood structure.  Colour and size can each be mapped to any
        node attribute stored in the graph.

        Parameters
        ----------
        color : str, optional
            Node attribute used for colouring.  Integer attributes with
            ≤ 20 unique values (e.g. ``'label'``, ``'epoch_created'``) are
            treated as **categorical** and produce a colour legend.  Float
            attributes (e.g. ``'density'``, ``'error'``,
            ``'average_distance'``) are treated as **continuous** and
            produce a colorbar.  When ``color='label'`` and the estimator
            has a ``classes_`` attribute the legend shows the original class
            names instead of integer indices.

        size : str, optional
            Node attribute used for sizing (e.g. ``'hit_count'``).  Each
            node's area in points² is ``min_size + value * scale_factor``.
            A size legend with three representative values is added.

        layout : {'spring_weighted', 'spring', 'grid', 'pca'}, \
default = 'spring_weighted'
            Algorithm used to compute node positions.

            ``'spring_weighted'``
                Force-directed layout where the spring constant of each edge
                is proportional to ``weight_distance`` (= 1 / Euclidean
                distance between adjacent weight vectors).  Neurons with
                similar weight vectors are pulled together, directly
                reflecting the U-matrix structure.

            ``'spring'``
                Unweighted Fruchterman–Reingold force-directed layout.

            ``'grid'``
                Neurons are placed at their integer SOM grid coordinates.
                Preserves the topological map structure but ignores
                feature-space distances.

            ``'pca'``
                The weight vectors are projected to 2-D with PCA.  Node
                positions reflect the principal directions of variance in
                feature space.

        seed : int or None, default = 0
            Random seed for the spring layouts.  Set to ``None`` for a
            non-deterministic result.

        min_size : float, default = 2
            Minimum node area in points².

        scale_factor : float, default = 1
            Multiplier applied to the *size* attribute value before adding
            *min_size*.

        palette : str, default = ``'tab10'``
            Matplotlib colormap name.  Applied to both categorical and
            continuous colour mappings.

        ax : matplotlib.axes.Axes, optional
            Axes to draw on.  A new figure is created when *None*.

        figsize : tuple of int, default = ``(10, 8)``
            Figure size in inches.  Ignored when *ax* is provided.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes the graph was drawn on.

        Examples
        --------
        >>> clf.plot_graph(color="label", size="hit_count")
        >>> clf.plot_graph(color="density", layout="pca", palette="magma_r")
        >>> clf.plot_graph(color="label", layout="grid")

        """
        check_is_fitted(self)

        nodes = list(self.som_.nodes)
        pos = self._compute_graph_layout(layout, nodes, seed)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        raw_sizes, node_sizes = self._compute_node_sizes(
            nodes, size, min_size, scale_factor
        )

        nx.draw_networkx_edges(self.som_, pos, ax=ax, alpha=1, edge_color="gray")
        self._draw_colored_nodes(ax, pos, nodes, node_sizes, color, palette)

        if size is not None and raw_sizes is not None:
            self._draw_size_legend(ax, raw_sizes, size, min_size, scale_factor)

        ax.axis("off")
        plt.tight_layout()
        return ax

    def _compute_graph_layout(
        self,
        layout: str,
        nodes: list,
        seed: int | None,
    ) -> dict:
        """Return a ``{node: (x, y)}`` position dict for the given layout strategy."""
        if layout == "grid":
            return {n: n for n in nodes}
        if layout == "spring":
            return nx.spring_layout(self.som_, seed=seed)
        if layout == "spring_weighted":
            return nx.spring_layout(self.som_, weight="weight_distance", seed=seed)
        if layout == "pca":
            from sklearn.decomposition import PCA

            coords = PCA(n_components=2).fit_transform(self.weights_)
            return {n: coords[i] for i, n in enumerate(nodes)}
        raise ValueError(
            f"Unknown layout {layout!r}. "
            "Choose from 'spring_weighted', 'spring', 'grid', 'pca'."
        )

    def _compute_node_sizes(
        self,
        nodes: list,
        size: str | None,
        min_size: float,
        scale_factor: float,
    ) -> tuple[np.ndarray | None, dict]:
        """Return ``(raw_sizes_array_or_None, {node: point²})``."""
        if size is None:
            return None, {n: float(min_size) for n in nodes}
        raw = np.array([self.som_.nodes[n].get(size, 0) for n in nodes], dtype=float)
        sized = {
            n: float(min_size + raw[i] * scale_factor) for i, n in enumerate(nodes)
        }
        return raw, sized

    def _draw_colored_nodes(
        self,
        ax: "plt.Axes",
        pos: dict,
        nodes: list,
        node_sizes: dict,
        color: str | None,
        palette: str,
    ) -> None:
        """Draw nodes with colour encoding; add legend or colorbar as appropriate."""
        draw_kwargs = dict(
            ax=ax,
            edgecolors="black",
            linewidths=0.5,
        )
        sizes = [node_sizes[n] for n in nodes]

        if color is None:
            nx.draw_networkx_nodes(
                self.som_,
                pos,
                nodelist=nodes,
                node_size=sizes,
                node_color="steelblue",
                **draw_kwargs,
            )
            return

        raw_vals = [self.som_.nodes[n].get(color) for n in nodes]
        sample = raw_vals[0] if raw_vals else 0
        is_categorical = isinstance(sample, (str, bool)) or (
            isinstance(sample, (int, np.integer)) and len(set(raw_vals)) <= 20
        )

        if is_categorical:
            self._draw_categorical_nodes(
                ax, pos, nodes, node_sizes, raw_vals, color, palette, draw_kwargs
            )
        else:
            sc = nx.draw_networkx_nodes(
                self.som_,
                pos,
                nodelist=nodes,
                node_size=sizes,
                node_color=np.array(raw_vals, dtype=float),
                cmap=plt.get_cmap(palette),
                **draw_kwargs,
            )
            plt.colorbar(sc, ax=ax, label=color, shrink=0.7)

    def _draw_categorical_nodes(
        self,
        ax: "plt.Axes",
        pos: dict,
        nodes: list,
        node_sizes: dict,
        raw_vals: list,
        color: str,
        palette: str,
        draw_kwargs: dict,
    ) -> None:
        """Draw one group of nodes per category value and add a colour legend."""
        unique_vals = sorted(set(raw_vals))
        cmap = plt.get_cmap(palette)
        color_map = {
            v: cmap(i / max(len(unique_vals) - 1, 1)) for i, v in enumerate(unique_vals)
        }

        for val in unique_vals:
            nodelist = [n for n, v in zip(nodes, raw_vals) if v == val]
            if not nodelist:
                continue
            if color == "label" and hasattr(self, "classes_"):
                legend_label = str(self.classes_[int(val)]) if val >= 0 else "dead"
            else:
                legend_label = str(val)
            nx.draw_networkx_nodes(
                self.som_,
                pos,
                nodelist=nodelist,
                node_size=[node_sizes[n] for n in nodelist],
                node_color=[color_map[val]],
                label=legend_label,
                **draw_kwargs,
            )

        color_legend = ax.legend(
            loc="upper left", title=color, scatterpoints=1, frameon=True
        )
        ax.add_artist(color_legend)

    def _draw_size_legend(
        self,
        ax: "plt.Axes",
        raw_sizes: np.ndarray,
        size: str,
        min_size: float,
        scale_factor: float,
    ) -> None:
        """Add a size legend with three representative values."""
        rep_vals = np.linspace(float(raw_sizes.min()), float(raw_sizes.max()), 3)
        handles = [
            ax.scatter(
                [],
                [],
                s=min_size + v * scale_factor,
                color="gray",
                alpha=0.6,
                edgecolors="black",
            )
            for v in rep_vals
        ]
        ax.legend(
            handles,
            [f"{v:.2g}" for v in rep_vals],
            loc="lower left",
            title=size,
            frameon=True,
        )

    def _get_u_matrix(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Calculate the average distance from each neuron to its neighbors in the input space."""  # noqa: E501
        g = self.som_
        node_to_idx = {node: i for i, node in enumerate(self.neurons_)}
        weights = self.weights_

        src, dst = zip(
            *(
                (node_to_idx[n], node_to_idx[nb])
                for n in self.neurons_
                for nb in g.neighbors(n)
            )
        )
        src = np.array(src)
        dst = np.array(dst)

        edge_distances = np.linalg.norm(weights[src] - weights[dst], axis=1)
        n = len(self.neurons_)
        counts = np.bincount(src, minlength=n)
        total = np.bincount(src, weights=edge_distances, minlength=n)
        return np.where(counts > 0, total / counts, 0.0)

    # def _calculate_rep(self, X: npt.NDArray) -> None:
    #     """Return the resemble entropy parameter.

    #     1. Calculate histogram of components of each sample.
    #     2. Calculate entropy of each sample from histogram
    #     3. Save minimum and maximum rep for all classes

    #     Use 20 bins as default
    #     """
    #     hists = []
    #     for sample in X:
    #         hists.append(np.histogram(sample, bins=20)[0])

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
        self._total_variance = np.var(data, axis=0).sum()
        self._neurons_added = True

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

    def _grow_som(self, data: npt.NDArray, y: npt.NDArray | None) -> None:
        """Second training phase."""
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
            if self._neurons_added:
                self.neurons_ = list(self.som_.nodes)
                self._distance_matrix = nx.floyd_warshall_numpy(self.som_)

            distances, winners = self._get_winning_neurons(data, n_bmu=1)
            sample_weights = self._calculate_exp_similarity(distances)

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
        """Create a graph containing the first four neurons in a square. Each neuron has a weight vector randomly chosen from the training samples."""  # noqa: E501
        n_samples = data.shape[0]
        chosen_indices = self.random_state_.choice(n_samples, size=4, replace=False)
        init_vectors = data[chosen_indices]
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
        self, sample_weights: npt.NDArray, winners: npt.NDArray, data: npt.NDArray
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

    def _calculate_gaussian_neighborhood(self) -> npt.NDArray:
        """Calculate the gaussian neighborhood function for all neuron
        pairs using the distance matrix.
        """
        sigma = self._calculate_current_sigma()
        h = np.exp(-(self._distance_matrix**2 / (2 * sigma**2)))

        return h

    def _calculate_exp_similarity(self, distances: np.ndarray) -> npt.NDArray:
        """Calculate the weight of each sample by calculating a exponential kernel
        for the distance between the sample and the bmu.
        """
        gamma = self._total_variance**-1
        kernel = 1 - (1 - np.exp(-gamma * distances**2)) ** 0.5
        return kernel

    # @profile
    def _write_accumulative_error(
        self, winners: np.ndarray, y: npt.NDArray | None, distances: np.ndarray
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
        """Distributes the error values of neurons in the SOM which are not boundary
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
        self._neurons_added = True

        for i in sorted_indices:
            node = list(self.som_.nodes)[i]
            node_degree = nx.degree(self.som_, node)
            if error_values[i] > self.growing_threshold_ and node_degree < 4:
                new_node, new_weight = self._insert_neuron(node)
                self._add_node_to_graph(node=new_node, weight=new_weight)
            else:
                break

    def _insert_neuron(self, bo: tuple[int, int]) -> tuple[tuple[int, int], np.ndarray]:
        """Insert one new neuron around the boundary neuron bo.

        Implements the directed insertion rule from Section 3.3.1.1 of the paper:
        prefer the free position whose adjacent existing neuron (corner neighbor)
        has the highest accumulative error. If a free position has no corner
        neighbor, fall back to the error of the neuron directly opposite bo.

        Weight is initialized by reflecting the opposite neighbor through bo
        (rule 1w / 2w / 3w), then averaged with the corner neighbor if one exists.
        """
        bx, by = bo
        all_adjacent = [(bx + 1, by), (bx - 1, by), (bx, by + 1), (bx, by - 1)]
        free_positions = [p for p in all_adjacent if p not in self.som_.nodes]

        def corner_neighbors_of(p: tuple[int, int]) -> list[tuple[int, int]]:
            """Existing grid neighbors of p, excluding bo itself."""
            px, py = p
            return [
                n
                for n in [(px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)]
                if n in self.som_.nodes and n != bo
            ]

        def opposite_of(p: tuple[int, int]) -> tuple[int, int] | None:
            """Existing neighbor of bo directly opposite p, or any neighbor as fallback."""  # noqa: E501
            op = (2 * bx - p[0], 2 * by - p[1])
            if op in self.som_.nodes:
                return op
            return next(iter(self.som_.neighbors(bo)), None)

        best_score = -1.0
        best_p = free_positions[0]
        best_corner: tuple[int, int] | None = None

        for p in free_positions:
            corners = corner_neighbors_of(p)
            if corners:
                corner = max(corners, key=lambda n: self.som_.nodes[n]["error"])
                score = self.som_.nodes[corner]["error"]
            else:
                op = opposite_of(p)
                score = self.som_.nodes[op]["error"] if op else 0.0
                corner = None

            if score > best_score:
                best_score = score
                best_p = p
                best_corner = corner

        # Weight initialization: reflect opposite neighbor through bo, then
        # average with the corner neighbor when one guides the position choice.
        op = opposite_of(best_p)
        w_bo = self.som_.nodes[bo]["weight"]
        w_base = (2 * w_bo - self.som_.nodes[op]["weight"]) if op else w_bo.copy()

        if best_corner is not None:
            w_new = (w_base + self.som_.nodes[best_corner]["weight"]) / 2
        else:
            w_new = w_base

        return best_p, w_new

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
        node, if exist.
        """
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
        """Return the average distance from each sample to the nearest prototype.

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
        X = validate_data(self, X, reset=False)
        distances, _ = self._get_winning_neurons(X, n_bmu=1)
        error = float(np.mean(distances))
        return error

    def _calculate_topographic_error(self, X: npt.NDArray) -> float:
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

    def topographic_function(self, X: npt.ArrayLike) -> npt.NDArray:
        """Compute the topographic function for the SOM.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Data used to compute the topographic function.

        Returns
        -------
        ndarray of shape (2, 2 * max_dist)
            Topographic function values for negative and positive distances.

        """
        X = validate_data(self, X)
        self._delaunay_matrix = self._calculate_delaunay_triangulation(X)

        # 1. Nur die wirklich benötigten Matrizen berechnen (Manhattan gelöscht)
        self.euclid_dist_matrix = euclidean_distances(self.neurons_)
        self.max_dist_matrix = pairwise_distances(self.neurons_, metric="chebyshev")

        max_dist = int(self.max_dist_matrix.max())
        k_values = np.arange(-max_dist, max_dist + 1)
        phi_values = np.zeros(len(k_values), dtype=np.float64)

        # 2. ÜBERRAGENDER PERFORMANCE-TRICK: Vorfiltern der Matrizen in 1D-Arrays
        # Für k > 0 interessieren uns nur max_dist-Werte, wo Delaunay == 1 ist
        valid_max_dists = self.max_dist_matrix[self._delaunay_matrix == 1]

        # Für k < 0 interessieren uns nur Delaunay-Werte, wo Euclid == 1 ist
        valid_delaunay_dists = self._delaunay_matrix[self.euclid_dist_matrix == 1]

        # 3. Werte für k berechnen (ohne teure Matrix-Scans)
        for idx, k in enumerate(k_values):
            if k > 0:
                phi_values[idx] = np.sum(valid_max_dists > k)
            elif k < 0:
                phi_values[idx] = np.sum(valid_delaunay_dists > -k)

        # 4. Sonderfall k = 0 berechnen (_phi(-1) + _phi(1))
        # Wir suchen die Indizes in unserem fertigen phi_values Array
        idx_neg1 = np.where(k_values == -1)[0]
        idx_pos1 = np.where(k_values == 1)[0]
        idx_zero = np.where(k_values == 0)[0]

        if len(idx_zero) > 0:
            val_neg1 = phi_values[idx_neg1[0]] if len(idx_neg1) > 0 else 0.0
            val_pos1 = phi_values[idx_pos1[0]] if len(idx_pos1) > 0 else 0.0
            phi_values[idx_zero[0]] = val_neg1 + val_pos1

        # Normierung der X-Achse
        normalized_distances = k_values / max_dist if max_dist > 0 else k_values

        return np.vstack((phi_values, normalized_distances))

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
    learning_rate: None = None,
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
    """Calculate the centers of the Voronoi regions based on the winners and data arrays."""
    voronoi_set_centers = np.zeros(shape=shape)
    for i in nb.prange(groups.size):
        group_start = offsets[i]
        group_end = offsets[i + 1] if i + 1 < groups.size else index.size
        group_index = index[group_start:group_end]
        samples = data[group_index]
        weights = kernel[group_index]
        weight_sum = np.sum(weights)
        for j in nb.prange(samples.shape[1]):
            if weight_sum == 0.0:
                # All kernel weights underflowed to zero (very large distances
                # relative to the data variance). Fall back to an unweighted mean
                # so the weight vector still moves toward the Voronoi centroid.
                mean_samples = np.mean(samples[:, j])
            else:
                mean_samples = np.average(samples[:, j], weights=weights)
            voronoi_set_centers[i, j] = mean_samples

    return voronoi_set_centers


@nb.njit(fastmath=True)
def numba_quantization_error(
    winners: npt.NDArray, length: int, distances: npt.NDArray
) -> npt.NDArray:
    """Berechnet den Quantisierungsfehler effizient mit np.bincount."""
    return np.bincount(winners, weights=distances, minlength=length)

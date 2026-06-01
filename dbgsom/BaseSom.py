"""Handles the core SOM functionality."""

import sys
from abc import ABC, abstractmethod
from math import exp, log, sqrt
from numbers import Integral, Real
from typing import Any, Callable, Self

from sklearn.utils.validation import validate_data

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
    from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

    # from line_profiler import profile
    from sklearn.preprocessing import normalize
    from sklearn.utils import check_random_state
    from sklearn.utils._param_validation import Interval, StrOptions
    from sklearn.utils.multiclass import check_classification_targets
    from sklearn.utils.validation import check_is_fitted
    from tqdm import tqdm
except ImportError as e:
    print(e)
    sys.exit()


class BaseSom(BaseEstimator, ABC):
    """Base Self-Organizing Map (SOM) implementation.

    A Self-Organizing Map is an unsupervised neural network model that maps
    high-dimensional input data onto a low-dimensional, typically 2D or 3D
    grid of neurons. The neurons are updated during training to preserve the
    topological properties of the input space.

    Parameters
    ----------
    n_iter : int, default=500
        Maximum number of iterations for training.

    convergence_iter : int, default=50
        Number of epochs without growth before a convergence check triggers
        the next growth step or the transition to fine training.

    spreading_factor : float, default=0.5
        Factor controlling the spread of neuron activation.

    sigma_start : float or None, default=None
        Initial standard deviation of the neighborhood function at the start
        of coarse training. Defaults to ``0.2 * sqrt(self.n_neurons)``

    sigma_end : float or None, default=None
        Target standard deviation at the end of coarse training. Defaults to
        ``0.05 * sqrt(self.n_neurons)`` (~5% of map length).

    sigma_fine : float or None, default=None
        Fixed standard deviation used throughout the fine training phase.
        If set to None is gets set to sigma_end.
        Small values (~0.1) concentrate updates on the BMU and minimise the
        quantization error, as recommended by Kohonen.

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

    convergence_threshold : float, default=1e-5
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

    tau_2 : float, default=0.5
        Global stopping criterion threshold for vertical growth (τ₂ in the GHSOM paper).
        A unit is expanded into a new child SOM when its quantization error exceeds
        ``tau_2 * qe_0``, where ``qe_0`` is the quantization error of a single unit
        whose weight equals the mean of all training data.

    n_jobs : int, default=1
        Number of parallel jobs for computation.

    """

    def __init__(
        self,
        n_iter: int = 500,
        convergence_iter: int = 50,
        spreading_factor: float = 0.5,
        sigma_start: float | None = None,
        sigma_end: float | None = None,
        sigma_fine: float | None = None,
        vertical_growth: bool = False,
        decay_function: str = "exponential",
        neighborhood_function: str = "gaussian",
        learning_rate: float = 0.02,
        verbose: bool = False,
        coarse_training_frac: float = 0.5,
        random_state: int | None | np.random.RandomState = None,
        convergence_threshold: float = 10**-5,
        max_neurons: int = 100,
        metric: str = "euclidean",
        threshold_method: str = "se",
        growth_criterion: str = "quantization_error",
        min_samples_vertical_growth: int = 100,
        tau_2: float = 0.5,
        n_jobs: int = 1,
    ) -> None:
        super().__init__()
        self.spreading_factor = spreading_factor
        self.n_iter = n_iter
        self.convergence_iter = convergence_iter
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.sigma_fine = sigma_fine
        self.decay_function = decay_function
        self.neighborhood_function = neighborhood_function
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.coarse_training_frac = coarse_training_frac
        self.random_state = random_state
        self.convergence_threshold = convergence_threshold
        self.max_neurons = max_neurons
        self.metric = metric
        self.threshold_method = threshold_method
        self.growth_criterion = growth_criterion
        self.min_samples_vertical_growth = min_samples_vertical_growth
        self.tau_2 = tau_2
        self.vertical_growth = vertical_growth
        self.n_jobs = n_jobs

    _parameter_constraints = {
        "n_iter": [Interval(Integral, 1, None, closed="left")],
        "convergence_iter": [Interval(Integral, 1, None, closed="left")],
        "max_neurons": [Interval(Integral, 4, None, closed="left")],
        "min_samples_vertical_growth": [Interval(Integral, 1, None, closed="left")],
        "spreading_factor": [Interval(Real, 0, 1, closed="neither")],
        "learning_rate": [Interval(Real, 0, None, closed="neither")],
        "coarse_training_frac": [Interval(Real, 0, 1, closed="neither")],
        "convergence_threshold": [Interval(Real, 0, None, closed="neither")],
        "sigma_start": [Interval(Real, 0, None, closed="neither"), None],
        "sigma_end": [Interval(Real, 0, None, closed="neither"), None],
        "sigma_fine": [Interval(Real, 0, None, closed="neither"), None],
        "decay_function": [StrOptions({"exponential", "linear"})],
        "neighborhood_function": [StrOptions({"gaussian", "cutgauss"})],
        "threshold_method": [StrOptions({"classical", "se"})],
        "growth_criterion": [StrOptions({"entropy", "quantization_error"})],
        "tau_2": [Interval(Real, 0, 1, closed="neither")],
        "metric": [StrOptions({"euclidean", "cosine"})],
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
            X = validate_data(self, X, dtype=None, ensure_min_samples=4)

        elif y is not None:
            X, y = validate_data(self, X, y, dtype=None, ensure_min_samples=4)
            check_classification_targets(y)
            classes, y = np.unique(y, return_inverse=True)
            self.classes_ = np.array(classes)
        if self.metric == "cosine":
            X = normalize(X)
        local_qe_0 = float(np.sum(np.linalg.norm(X - X.mean(axis=0), axis=1)))
        if not hasattr(self, "qe_0_"):
            self.qe_0_ = local_qe_0
        self.random_state_ = check_random_state(self.random_state)
        self._initialize_som(X)

        # Horizontal growing phase
        self._grow_som(X, y)
        # self.rep = self._calculate_rep(X)
        self.topographic_error_ = self._calculate_topographic_error(X)
        distances, _ = self._get_winning_neurons(X, n_bmu=1)
        self.quantization_error_ = float(np.mean(distances))
        self.n_features_in_ = X.shape[1]
        self._write_node_statistics(X)
        self._write_edge_statistics()
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

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    def _grow_vertical(self, X: npt.NDArray, y: None | npt.NDArray = None) -> None:
        """Triggers vertical growth in the SOM by creating new instances of the DBGSOM
        class and fitting them with filtered data.

        Reference: Qu et al., "Entropy-Defined Direct Batch Growing Hierarchical
        Self-Organizing Mapping for Efficient Network Anomaly Detection",
        IEEE Access, 2021.
        """
        # todo: refactor in sub classes
        self.vertical_growing_threshold_ = self.tau_2 * self.qe_0_
        _, winners = self._get_winning_neurons(X, n_bmu=1)
        node_to_idx = {node: i for i, node in enumerate(self.neurons_)}
        relevant_nodes = [
            node
            for (node, error) in self.som_.nodes(data="error")
            if error > self.vertical_growing_threshold_
        ]
        for node in relevant_nodes:
            new_som = clone(self)
            new_som.qe_0_ = self.qe_0_
            node_idx = node_to_idx[node]
            X_filtered = X[winners == node_idx]
            if y is not None:
                y_filtered = y[winners == node_idx]
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
            weight_x = som.nodes[u]["weight"]
            weight_y = som.nodes[v]["weight"]
            distance = np.linalg.norm(weight_x - weight_y)
            som.edges[u, v]["weight_distance"] = 1 / float(distance + 0.001)

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

        Reference: Teuvo Kohonen, "Description of Input Patterns by
        Linear Mixtures of SOM Models", Proceedings of the 6th International
        Workshop on Self-Organizing Maps, 2007.

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
        color: str | None = None,
        pointsize: str | None = None,
        layout: str = "grid",
        palette: str = "magma_r",
    ) -> so.Plot:
        """Plot the SOM neurons and their neighbourhood edges using seaborn objects.

        Edges are drawn first as grey lines; nodes are drawn on top and can be
        colour- and size-coded by any node attribute stored in the graph.

        Parameters
        ----------
        color : {'label', 'epoch_created', 'error', 'average_distance', 'density',
                 'hit_count', 'pca_rgb'}, optional
            Node attribute mapped to colour.  Numeric attributes with all
            identical values are cast to string to avoid a degenerate
            continuous scale.

            Pass ``'pca_rgb'`` to colour each neuron by its position in the
            first three principal components of the weight space: PC1 -> R,
            PC2 -> G, PC3 -> B (each component min-max normalised to 0-255).
            Similar colours indicate similar weight vectors; the colour pattern
            reveals the topological structure of the feature space.

        pointsize : {'label', 'epoch_created', 'error', 'average_distance',
                     'density', 'hit_count'}, optional
            Node attribute mapped to point size.

        layout : {'grid', 'pca'}, default 'grid'
            Algorithm used to compute node positions.

            ``'grid'``
                Neurons are placed at their integer SOM grid coordinates.
                Preserves the topological map structure.
            ``'pca'``
                Weight vectors projected to 2-D with PCA.  Node positions
                reflect the principal directions of variance in feature space.

        palette : str, default ``'magma_r'``
            Seaborn / Matplotlib colormap name applied to the colour mapping.

        """
        check_is_fitted(self)

        nodes = list(self.som_.nodes)
        pos = self._compute_graph_layout(layout, nodes)

        # -- Node DataFrame ---------------------------------------------------
        node_data = pd.DataFrame(dict(self.som_.nodes)).T.reset_index(drop=True)
        coords = pd.DataFrame(
            [(pos[n][0], pos[n][1]) for n in nodes], columns=["x", "y"]
        )
        nodes_df = pd.concat([coords, node_data], axis=1)

        for col in (
            "epoch_created",
            "error",
            "density",
            "hit_count",
            "average_distance",
        ):
            if col in nodes_df.columns:
                nodes_df[col] = pd.to_numeric(nodes_df[col], errors="raise")

        if "label" in nodes_df.columns:
            if hasattr(self, "classes_"):
                classes = self.classes_
                nodes_df["label"] = nodes_df["label"].apply(
                    lambda i: str(classes[int(i)]) if int(i) >= 0 else "dead"
                )
            else:
                nodes_df["label"] = nodes_df["label"].astype(str)

        # -- Edge DataFrame (NaN-separated path) ------------------------------
        # NaN sentinels are more reliable than group= for splitting segments;
        # matplotlib breaks a Path at every NaN row.
        edge_rows: list[dict] = []
        for u, v in self.som_.edges():
            edge_rows.append({"x": float(pos[u][0]), "y": float(pos[u][1])})
            edge_rows.append({"x": float(pos[v][0]), "y": float(pos[v][1])})
            edge_rows.append({"x": np.nan, "y": np.nan})  # segment break
        edges_df = (
            pd.DataFrame(edge_rows) if edge_rows else pd.DataFrame(columns=["x", "y"])
        )

        # -- Aesthetic mappings -----------------------------------------------
        node_aesthetics: dict[str, str] = {}

        color_col, color_scale = self._resolve_color_aesthetic(nodes_df, color, palette)
        if color_col is not None:
            node_aesthetics["color"] = color_col

        if isinstance(pointsize, str) and pointsize in nodes_df.columns:
            if (
                pd.api.types.is_numeric_dtype(nodes_df[pointsize])
                and nodes_df[pointsize].nunique() <= 1
            ):
                nodes_df[pointsize] = nodes_df[pointsize].astype(str)
            node_aesthetics["pointsize"] = pointsize

        # -- Build and show plot ----------------------------------------------
        p = so.Plot(nodes_df, x="x", y="y")

        if not edges_df.empty:
            p = p.add(
                so.Path(color="gray", linewidth=0.5),
                data=edges_df,
            )

        p = p.add(so.Dot(), data=nodes_df, **node_aesthetics)

        if color_scale is not None:
            p = p.scale(color=color_scale)

        return p

    def _resolve_color_aesthetic(
        self,
        nodes_df: pd.DataFrame,
        color: str | None,
        palette: str,
    ) -> tuple[str | None, "so.Scale | None"]:
        """Resolve the colour aesthetic and build the matching seaborn scale.

        Returns
        -------
        col_name : str or None
            Column name in *nodes_df* to use as the ``color`` aesthetic,
            or ``None`` when no colour mapping is requested.
        scale : so.Scale or None
            Ready-to-use scale object, or ``None`` when *col_name* is ``None``.

        """
        if color == "pca_rgb":
            hex_colors = self._compute_pca_rgb_colors()
            nodes_df["pca_rgb"] = hex_colors
            return "pca_rgb", so.Nominal(values={c: c for c in hex_colors})

        if color is not None and color in nodes_df.columns:
            if (
                pd.api.types.is_numeric_dtype(nodes_df[color])
                and nodes_df[color].nunique() <= 1
            ):
                nodes_df[color] = nodes_df[color].astype(str)
            return color, self._build_color_scale(nodes_df[color], palette)

        return None, None

    def _build_color_scale(
        self,
        series: pd.Series,
        palette: str,
    ) -> "so.Scale":
        """Return the appropriate seaborn objects colour scale for *series*.

        Numeric series -> ``so.Continuous`` with *palette* as a matplotlib
        colormap.  Categorical / string series -> ``so.Nominal`` populated
        with colours drawn from the named seaborn palette.

        """
        if pd.api.types.is_numeric_dtype(series):
            return so.Continuous(palette)
        import seaborn as sns

        colors = sns.color_palette(palette, n_colors=series.nunique())
        return so.Nominal(values=list(colors))

    def _compute_pca_rgb_colors(self) -> list[str]:
        """Project weight vectors to 3-D with PCA and map to RGB hex strings.

        Each of the three principal components is independently min-max
        normalised to [0, 255] and mapped to the R, G, and B channel
        respectively.  Neurons with similar weight vectors receive similar
        colours; the resulting colour pattern reveals the topological
        structure of the feature space.

        Returns
        -------
        hex_colors : list of str
            One ``'#rrggbb'`` string per neuron, in the same order as
            ``self.neurons_``.

        """
        from sklearn.decomposition import PCA

        n_components = min(3, self.weights_.shape[1], self.weights_.shape[0])
        components = PCA(n_components=n_components).fit_transform(self.weights_)

        # Min-max normalise each component independently to [0, 1]
        col_min = components.min(axis=0)
        col_max = components.max(axis=0)
        col_range = np.where(col_max - col_min == 0, 1.0, col_max - col_min)
        components_norm = (components - col_min) / col_range

        # Pad to exactly 3 channels if the data has fewer than 3 features
        if n_components < 3:
            pad = np.zeros((len(components_norm), 3 - n_components))
            components_norm = np.hstack([components_norm, pad])

        return [
            "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
            for r, g, b in components_norm
        ]

    def _compute_graph_layout(self, layout: str, nodes: list) -> dict:
        """Return a ``{node: (x, y)}`` position dict for the given layout strategy."""
        if layout == "grid":
            return {n: n for n in nodes}
        if layout == "pca":
            from sklearn.decomposition import PCA

            coords = PCA(n_components=2).fit_transform(self.weights_)
            return {n: coords[i] for i, n in enumerate(nodes)}
        raise ValueError(f"Unknown layout {layout!r}. Choose from 'grid', 'pca'.")

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
        self._last_growth_epoch = -1
        self._sigma_coarse: float | None = None
        self._training_phase = "coarse"
        self.growing_threshold_ = self._calculate_growing_threshold(data)
        self._total_variance = np.var(data, axis=0).sum()
        self._neurons_added = True

        self.som_ = self._create_som(data)
        self._distance_matrix = nx.floyd_warshall_numpy(self.som_)
        self.weights_ = self._extract_values_from_graph("weight")
        self.neurons_ = list(self.som_.nodes)

    def _calculate_growing_threshold(self, data: npt.NDArray) -> float:
        """Compute the growing threshold for neuron insertion.

        The classical method uses ``-n_dim * log(spreading_factor)``.
        The statistics-enhanced method (``threshold_method="se"``) scales by
        the norm of the data standard deviation for data-adaptive thresholding.

        References
        ----------
        Vasighi and Amini, "A directed batch growing approach to enhance the
        topology preservation of self-organizing map", Applied Soft Computing,
        2017.

        Qu et al., "Statistics-enhanced Direct Batch Growth Self-Organizing
        Mapping for efficient DoS Attack Detection", IEEE Access, 2019.

        """
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
        """Second training phase: iterative weight update and neuron insertion.

        Reference: Vasighi and Amini, "A directed batch growing approach to
        enhance the topology preservation of self-organizing map", Applied Soft
        Computing, 2017.
        """
        for current_epoch in tqdm(
            iterable=range(self.n_iter),
            disable=not self.verbose,
            unit=" epochs",
        ):
            print(self._calculate_current_sigma())
            self._current_epoch = current_epoch
            if current_epoch > self.coarse_training_frac * self.n_iter:
                self._training_phase = "fine"
            # check if new neurons were inserted
            if self._neurons_added:
                self.neurons_ = list(self.som_.nodes)
                self._distance_matrix = nx.floyd_warshall_numpy(self.som_)
                self.weights_ = self._extract_values_from_graph("weight")
                self._neurons_added = False

            distances, winners = self._get_winning_neurons(data, n_bmu=1)
            sample_weights = self._calculate_exp_similarity(distances)

            self._update_weights(sample_weights, winners, data)
            self._write_accumulative_error(winners, y, distances)
            if self.converged_ and self._training_phase == "fine":
                break

            epochs_since_growth = current_epoch - self._last_growth_epoch
            if (
                self._training_phase == "coarse"
                and len(self.neurons_) < self.max_neurons
                and (self.converged_ or epochs_since_growth >= self.convergence_iter)
            ):
                converged_triggered = self.converged_
                self._distribute_errors()
                self._add_new_neurons()
                self.converged_ = False
                self._last_growth_epoch = current_epoch
                self._sigma_coarse = self._compute_decayed_sigma(current_epoch)
                if converged_triggered and not self._neurons_added:
                    self._training_phase = "fine"

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
        """Return distances and indices of the n_bmu nearest prototypes per sample.

        Uses a numba-JIT kernel for n_bmu=1 (hot path: fused distance + argmin/argmax,
        no distance matrix allocated). Falls back to BLAS + argpartition for
        n_bmu > 1 (only used post-training for topographic error).
        """
        if self.metric == "cosine":
            data = normalize(data)
            if n_bmu == 1:
                return numba_find_winners_cosine(data, self.weights_)
            sim_matrix = data @ self.weights_.T
            dist_matrix = 1.0 - sim_matrix
            part = np.argpartition(dist_matrix, n_bmu, axis=1)[:, :n_bmu]
            row_idx = np.arange(len(data))[:, np.newaxis]
            order = np.argsort(dist_matrix[row_idx, part], axis=1)
            winners = part[row_idx, order]
            distances = dist_matrix[row_idx, winners]
            return distances, winners
        if n_bmu == 1:
            return numba_find_winners(data, self.weights_)
        dist_matrix = euclidean_distances(data, self.weights_)
        part = np.argpartition(dist_matrix, n_bmu, axis=1)[:, :n_bmu]
        row_idx = np.arange(len(data))[:, np.newaxis]
        order = np.argsort(dist_matrix[row_idx, part], axis=1)
        winners = part[row_idx, order]
        distances = dist_matrix[row_idx, winners]
        return distances, winners

    @abstractmethod
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

        # Step 2
        neuron_activations = np.bincount(winners, minlength=len(self.neurons_))

        # Step 3
        gaussian_kernel = self._calculate_gaussian_neighborhood()

        # Step 4 — weighted[i,j] = h[i,j] * n_j; contract over j via BLAS
        weighted = gaussian_kernel * neuron_activations
        numerator = weighted @ voronoi_set_centers
        denominator = weighted.sum(axis=1, keepdims=True)
        zero_denom = (denominator == 0).squeeze()
        safe_denom = np.where(denominator == 0, 1.0, denominator)
        new_weights = numerator / safe_denom
        new_weights[zero_denom] = self.weights_[zero_denom]

        # Step 5
        if self.metric == "cosine":
            new_weights = normalize(new_weights)
        if np.linalg.norm(self.weights_ - new_weights) < self.convergence_threshold:
            self.converged_ = True
        self.weights_ = new_weights
        nx.set_node_attributes(
            G=self.som_, values=dict(zip(self.neurons_, self.weights_)), name="weight"
        )

    def _calculate_gaussian_neighborhood(self) -> npt.NDArray:
        """Calculate the neighborhood function for all neuron pairs.

        "gaussian"  : standard Gaussian over graph distances
        "cutgauss"  : Gaussian with flanks set to zero beyond 2 * sigma
        """
        sigma = self._calculate_current_sigma()
        h = np.exp(-(self._distance_matrix**2 / (2 * sigma**2)))
        if self.neighborhood_function == "cutgauss":
            h *= self._distance_matrix <= 2 * sigma
        return h

    def _calculate_exp_similarity(self, distances: np.ndarray) -> npt.NDArray:
        """Calculate the weight of each sample via an exponential kernel on the
        BMU distance, downweighting outliers to improve robustness.

        Reference: D'Urso et al., "Smoothed self-organizing map for robust
        clustering", Information Sciences, 2019.
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

        The entropy growth criterion uses Shannon entropy of the class
        distribution per neuron instead of quantization error.

        Reference: Qu et al., "Entropy-Defined Direct Batch Growing Hierarchical
        Self-Organizing Mapping for Efficient Network Anomaly Detection",
        IEEE Access, 2021.
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

        Reference: Vasighi and Amini, "A directed batch growing approach to
        enhance the topology preservation of self-organizing map", Applied Soft
        Computing, 2017.
        """
        error_values = self._extract_values_from_graph("error")
        sorted_indices = np.argsort(-error_values)
        self._neurons_added = False
        nodes_list = list(self.som_.nodes)

        for i in sorted_indices:
            node = nodes_list[i]
            node_degree = nx.degree(self.som_, node)
            if error_values[i] > self.growing_threshold_ and node_degree < 4:
                new_node, new_weight = self._insert_neuron(node)
                self._add_node_to_graph(node=new_node, weight=new_weight)
                self._neurons_added = True
            else:
                break

    def _corner_neighbors_of(
        self, boundary_node: tuple[int, int], p: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """Return existing grid neighbors of p, excluding boundary_node itself."""
        px, py = p
        return [
            n
            for n in [(px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)]
            if n in self.som_.nodes and n != boundary_node
        ]

    def _opposite_of(
        self, boundary_node: tuple[int, int], p: tuple[int, int]
    ) -> tuple[int, int] | None:
        """Return the neighbor of boundary_node directly opposite p.

        Falls back to any neighbor of boundary_node when no exact opposite exists.
        """
        bx, by = boundary_node
        op = (2 * bx - p[0], 2 * by - p[1])
        if op in self.som_.nodes:
            return op
        return next(iter(self.som_.neighbors(boundary_node)), None)

    def _find_insertion_position(
        self,
        boundary_node: tuple[int, int],
        free_positions: list[tuple[int, int]],
    ) -> tuple[tuple[int, int], tuple[int, int] | None]:
        """Select the free position with the highest corner-neighbor error.

        Implements the directed insertion rule from Section 3.3.1.1.
        Falls back to the error of the opposite neighbor when no corner neighbor exists.

        Reference: Vasighi and Amini, "A directed batch growing approach to
        enhance the topology preservation of self-organizing map", Applied Soft
        Computing, 2017.
        """
        best_score = -1.0
        best_p = free_positions[0]
        best_corner: tuple[int, int] | None = None

        for p in free_positions:
            corners = self._corner_neighbors_of(boundary_node, p)
            if corners:
                corner = max(corners, key=lambda n: self.som_.nodes[n]["error"])
                score = self.som_.nodes[corner]["error"]
            else:
                op = self._opposite_of(boundary_node, p)
                score = self.som_.nodes[op]["error"] if op else 0.0
                corner = None

            if score > best_score:
                best_score = score
                best_p = p
                best_corner = corner

        return best_p, best_corner

    def _initialize_neuron_weight(
        self,
        boundary_node: tuple[int, int],
        best_p: tuple[int, int],
        best_corner: tuple[int, int] | None,
    ) -> np.ndarray:
        """Compute the initial weight for a new neuron at best_p.

        Reflects the opposite neighbor through boundary_node (rules 1w/2w/3w),
        then averages with the corner neighbor when one guided the position choice.

        Reference: Vasighi and Amini, "A directed batch growing approach to
        enhance the topology preservation of self-organizing map", Applied Soft
        Computing, 2017.
        """
        op = self._opposite_of(boundary_node, best_p)
        w_boundary = self.som_.nodes[boundary_node]["weight"]
        if op is not None:
            w_base = 2 * w_boundary - self.som_.nodes[op]["weight"]
        else:
            w_base = w_boundary.copy()

        if best_corner is not None:
            return (w_base + self.som_.nodes[best_corner]["weight"]) / 2
        return w_base

    def _insert_neuron(
        self, boundary_node: tuple[int, int]
    ) -> tuple[tuple[int, int], np.ndarray]:
        """Insert one new neuron around boundary_node."""
        bx, by = boundary_node
        all_adjacent = [(bx + 1, by), (bx - 1, by), (bx, by + 1), (bx, by - 1)]
        free_positions = [p for p in all_adjacent if p not in self.som_.nodes]
        best_p, best_corner = self._find_insertion_position(
            boundary_node, free_positions
        )
        w_new = self._initialize_neuron_weight(boundary_node, best_p, best_corner)
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
        """Return the neighborhood bandwidth for the current epoch.

        Coarse phase: returns ``_sigma_coarse``, which is reset to a
        decayed value via ``_compute_decayed_sigma`` after each growth step.
        Defaults to ``0.2 * sqrt(n_neurons)`` at the start.

        Fine phase: returns ``sigma_fine`` if set, otherwise 0.1.

        Returns
        -------
        float
            Neighborhood bandwidth for the current epoch.

        """
        n_neurons = self.som_.number_of_nodes()
        sigma_start = (
            0.2 * (sqrt(n_neurons) - 1)
            if self.sigma_start is None
            else self.sigma_start
        )

        if self._training_phase == "coarse":
            if self._sigma_coarse is None:
                self._sigma_coarse = sigma_start
            return self._sigma_coarse
        else:
            return self._sigma_coarse if self.sigma_fine is None else 0.1

    def _compute_decayed_sigma(self, epoch: int) -> float:
        """Return the new ``_sigma_coarse`` value after a growth step.

        Called once per growth step. Applies the configured decay function
        (exponential or linear) to interpolate between ``sigma_start`` and
        ``sigma_end`` based on the current epoch, so the coarse neighborhood
        shrinks progressively as the map grows.

        The epoch is normalized so that ``current_iter = n_iter`` at the end
        of the coarse phase (``epoch = coarse_training_frac * n_iter``).
        For exponential decay the learning rate is derived from ``n_iter`` so
        that exactly 99 % of the drop from ``sigma_start`` to ``sigma_end``
        is completed by that point: ``lr = log(100) / n_iter``.

        Parameters
        ----------
        epoch : int
            Current training epoch at the time of the growth step.

        Returns
        -------
        float
            Decayed sigma value to assign to ``_sigma_coarse``.

        """
        n_neurons = self.som_.number_of_nodes()
        sigma_start = (
            0.2 * sqrt(n_neurons) if self.sigma_start is None else self.sigma_start
        )
        sigma_end = 0.05 * sqrt(n_neurons) if self.sigma_end is None else self.sigma_end
        # lr chosen so exp(-lr * n_iter) = 0.01, i.e. 99 % decay at end of coarse phase
        normalized_lr = log(100) / self.n_iter
        decay_fn = _DECAY_FUNCTIONS[self.decay_function]
        return decay_fn(
            sigma_end=sigma_end,
            sigma_start=sigma_start,
            max_iter=self.n_iter,
            current_iter=epoch / self.coarse_training_frac,
            learning_rate=normalized_lr,
        )

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
        manhattan_dist_matrix = manhattan_distances(self.neurons_)
        topographic_error = 0
        for node in bmu_indices:
            distance = manhattan_dist_matrix[node[0], node[1]]
            topographic_error += 1 if distance > 1 else 0

        return topographic_error / X.shape[0]

    def topographic_function(self, X: npt.ArrayLike) -> npt.NDArray:
        """Compute the topographic function for the SOM.

        Measures topology preservation across all neighbourhood scales k.
        Positive k values detect fold-overs (map neighbours that are far apart
        in data space); negative k values detect tears (data neighbours that
        are far apart on the map). phi(0) = phi(-1) + phi(1).

        Reference: Villmann et al., "Topology preservation in self-organizing
        feature maps: exact definition and measurement", IEEE Trans. Neural
        Networks, 1997.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Data used to compute the topographic function.

        Returns
        -------
        ndarray of shape (2, 2 * max_dist + 1)
            Row 0: phi values; row 1: normalised k-axis in [-1, 1].

        """
        X = validate_data(self, X, reset=False)
        self._delaunay_matrix = self._calculate_delaunay_triangulation(X)

        # self._distance_matrix is the Floyd-Warshall graph distance on the SOM,
        # computed during fit(). It is the correct d_map for the topographic function.
        map_dist_matrix = self._distance_matrix

        max_dist = int(map_dist_matrix.max())
        k_values = np.arange(-max_dist, max_dist + 1)
        phi_values = np.zeros(len(k_values), dtype=np.float64)

        # Pre-filter to 1-D arrays to avoid repeated full-matrix scans in the loop.
        # For k > 0: map distances for pairs that are direct Delaunay neighbours.
        valid_map_dists = map_dist_matrix[self._delaunay_matrix == 1]
        # For k < 0: Delaunay distances for pairs that are direct SOM graph neighbours.
        valid_delaunay_dists = self._delaunay_matrix[map_dist_matrix == 1]

        for idx, k in enumerate(k_values):
            if k > 0:
                phi_values[idx] = np.sum(valid_map_dists > k)
            elif k < 0:
                phi_values[idx] = np.sum(valid_delaunay_dists > -k)

        # k=0 is defined as phi(-1) + phi(1).
        # Because k_values = arange(-max_dist, max_dist+1), k=0 sits at index max_dist.
        if max_dist >= 1:
            phi_values[max_dist] = phi_values[max_dist - 1] + phi_values[max_dist + 1]

        normalized_distances = k_values / max_dist if max_dist > 0 else k_values

        return np.vstack((phi_values, normalized_distances))

    def _calculate_delaunay_triangulation(self, X) -> np.ndarray:
        """Calculate the Delaunay triangulation distance matrix via the
        competitive Hebbian rule: connect BMU1 and BMU2 for each input sample.

        Reference: Villmann et al., "Topology preservation in self-organizing
        feature maps: exact definition and measurement", IEEE Trans. Neural
        Networks, 1997.
        """
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


_DECAY_FUNCTIONS: dict[str, Callable[..., float]] = {
    "linear": linear_decay,
    "exponential": exponential_decay,
}


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


@nb.njit(cache=True, parallel=True, fastmath=True)
def numba_find_winners_cosine(
    data: npt.NDArray, weights: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """Find the most similar weight vector per sample via fused dot-product + argmax.

    Assumes data and weights are already L2-normalised (unit vectors).
    No n×m similarity matrix is allocated.
    """
    n_samples = data.shape[0]
    n_features = data.shape[1]
    n_neurons = weights.shape[0]
    winners = np.empty(n_samples, dtype=np.int64)
    distances = np.empty(n_samples, dtype=np.float64)
    for i in nb.prange(n_samples):  # type: ignore[attr-defined]
        best_sim = -np.inf
        best_j = 0
        for j in range(n_neurons):
            sim = 0.0
            for k in range(n_features):
                sim += data[i, k] * weights[j, k]
            if sim > best_sim:
                best_sim = sim
                best_j = j
        winners[i] = best_j
        distances[i] = 1.0 - best_sim
    return distances, winners


@nb.njit(cache=True, parallel=True, fastmath=True)
def numba_find_winners(
    data: npt.NDArray, weights: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """Find the nearest weight vector for each sample (fused distance + argmin)."""
    n_samples = data.shape[0]
    n_features = data.shape[1]
    n_neurons = weights.shape[0]
    winners = np.empty(n_samples, dtype=np.int64)
    distances = np.empty(n_samples, dtype=np.float64)
    for i in nb.prange(n_samples):  # ty:ignore[not-iterable]
        best_dist_sq = np.inf
        best_j = 0
        for j in range(n_neurons):
            d_sq = 0.0
            for k in range(n_features):
                diff = data[i, k] - weights[j, k]
                d_sq += diff * diff
            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                best_j = j
        winners[i] = best_j
        distances[i] = np.sqrt(best_dist_sq)
    return distances, winners


@nb.njit(fastmath=True)
def numba_quantization_error(
    winners: npt.NDArray, length: int, distances: npt.NDArray
) -> npt.NDArray:
    """Berechnet den Quantisierungsfehler effizient mit np.bincount."""
    return np.bincount(winners, weights=distances, minlength=length)

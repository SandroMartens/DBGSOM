"""
Implements the SOM Clusterer."""

import numpy as np
import numpy.typing as npt
from sklearn.base import (
    ClusterMixin,
    TransformerMixin,
    check_array,
    check_is_fitted,
)

from .BaseSom import BaseSom


class SomVQ(BaseSom, ClusterMixin, TransformerMixin):
    """A Directed Batch Growing Self-Organizing Map.

    This class implements the vector quantization/clustering functionality of the SOM.

    Parameters
    ----------
    spreading_factor : float, default = 0.5
        Spreading factor to calculate the treshold for neuron insertion.

        0 < spreading_factor < 1.

        0 means no growth, 1 means unlimited growth.

    n_iter : int, default = 200
        Maximum Number of training epochs.

    convergence_iter : int, default = 1
        How many training iterations run until new neurons are added.

    max_neurons : int, default = 100
        Maximum number of neurons in the som.

    vertical_growth : bool, default = False
        Wether to trigger hierarchical growth.

    decay_function : {'exponential', 'linear'}, default = 'exponential'
        Decay function to use for neighborhood bandwith sigma.

    learning_rate : int, default = 0.02
        Decay factor if decay function is set to "exponential".

    verbose : bool, default = False

    coarse_training_frac : float, default = 0.5
        Fraction of max_iter to use for coarse training.

        Training happens in two phases, coarse and fine training. In coarse training,
        the neighborhood bandwidth is decreased from sigma_start to sigma_end and
        the network grows according to the growing rules. In fine training, the
        bandwidth is constant at sigma_end and no new neurons are added.

    growth_criterion : {"quantization_error", "entropy"}, default = "quantization_error"
        Method for calculating the error of neurons and samples.

        "quantization_error" : Use the quantization error of the prototypes.
        The cumulative error is the sum of individual errors of all samples.

        "entropy": For supervised learning we can use the entropy
        of labels of the samples represented by each prototype as error.

    metric : str, default = "euclidean"
        The metric to use for computing distances between prototypes and samples. Must
        be supported by scikit-learn or scipy.

    random_state : any (optional), default = None
        Random state for weight initialization.

    convergence_treshold : float, default = 10 ** -5
        If the sum of all weight changes is smaller than the threshold,
        convergence is assumed and the training is stopped.

    threshold_method : {"classical", "se"}, default = "se"
        Method to calculate the growing threshold.

        "classical" : Threshold is only dependent on the dimension of the input data.

        `gt =  -log(spreading_factor) * n_dim`

        "se" : Statistics enhanced formula, which uses the standard
        deviation of features in X.

        `gt = 150 * -log(spreading_factor) * np.sqrt(np.sum(np.std(X, axis=0) ** 2))`

    min_samples_vertical_growth : int, default = 100
        Minimum samples represented by a prototpye to trigger a vertical growth

    sigma_start, sigma_end : {None, numeric}, default = None
        Start and end value for the neighborhood bandwidth.

        If `None`, it is calculated dynamically in each epoch as

        `sigma_start = 0.2 * sqrt(n_neurons)`

        `sigma_end = max(0.7, 0.05 * sqrt(n_neurons))`

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Labels of each point.

    som_ : NetworkX.graph
        Graph object containing the neurons with attributes.

    weights_ : ndarray of shape (n_prototypes, n_features)
        Learned weights of the neurons

    topographic_error_ : float
        Fraction of training samples where the first and second best matching
        prototype are not neighbors on the SOM.

    quantization_error_ : float
        Average distance from all training samples to their nearest prototype.
    """

    def _check_input_data(self, X: npt.ArrayLike, y=None) -> tuple[npt.NDArray, None]:
        X = check_array(array=X, ensure_min_samples=4, dtype=[np.float64, np.float32])
        # throw away any y
        return X, None

    def _label_prototypes(self, X: npt.ArrayLike, y=None) -> None:
        for i, neuron in enumerate(self.som_):
            self.som_.nodes[neuron]["label"] = i

    def predict(self, X: npt.ArrayLike) -> np.ndarray:
        """Predict the closest neuron each sample in X belongs to.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            If fitted unsupervised: Index of best matching prototype.

        """
        check_is_fitted(self)
        X = check_array(X)
        _, labels = self._get_winning_neurons(X, n_bmu=1)

        return labels

    def _fit(self, X: npt.NDArray):

        self.labels_ = self.predict(X)

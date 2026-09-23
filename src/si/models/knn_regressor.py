from typing import Callable

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.models.knn_classifier import euclidean_distance


class KNNRegressor(Model):
    """
    K-Nearest Neighbors regressor.
    Estimates the label value for a sample based on the average value of the k most
    similar examples (shortest distance).

    Parameters
    ----------
    k: int
        The number of k nearest examples to consider.
    distance: Callable
        A function that calculates the distance between a sample and the samples in
        the training dataset (euclidean_distance by default).

    Estimated Parameters
    ---------------------
    dataset: Dataset
        Stores the training dataset.
    """

    def __init__(self, k: int = 5, distance: Callable = euclidean_distance, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Stores the training dataset.

        Parameters
        ----------
        dataset: Dataset
            The training dataset.

        Returns
        -------
        self: KNNRegressor
        """
        self.dataset = dataset
        return self

    def _get_closest_value(self, sample: np.ndarray) -> float:
        """
        Returns the average value of the k nearest examples to the given sample.

        Parameters
        ----------
        sample: numpy.ndarray (n_features,)
            A single sample.

        Returns
        -------
        float
            The estimated value for the sample.
        """
        distances = self.distance(sample, self.dataset.X)

        k_nearest_idxs = np.argsort(distances)[:self.k]
        k_nearest_values = self.dataset.y[k_nearest_idxs]

        return np.mean(k_nearest_values)

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Estimates the value for each sample in the testing dataset, based on
        the average value of the k most similar examples in the training dataset.

        Parameters
        ----------
        dataset: Dataset
            The testing dataset.

        Returns
        -------
        predictions: numpy.ndarray (n_samples,)
            An array of predicted values for the testing dataset.
        """
        return np.apply_along_axis(self._get_closest_value, axis=1, arr=dataset.X)

    def _score(self, dataset: Dataset) -> float:
        """
        Calculates the rmse between the estimated values and the actual ones.

        Parameters
        ----------
        dataset: Dataset
            The testing dataset.

        Returns
        -------
        error: float
            The rmse between predictions and actual values.
        """
        predictions = self._predict(dataset)
        return rmse(dataset.y, predictions)

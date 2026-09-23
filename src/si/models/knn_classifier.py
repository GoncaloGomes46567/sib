from typing import Callable

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the euclidean distance between a single sample x and every sample in y.

    Parameters
    ----------
    x: numpy.ndarray (n_features,)
        A single sample.
    y: numpy.ndarray (n_samples, n_features)
        Multiple samples.

    Returns
    -------
    numpy.ndarray (n_samples,)
        The distance between x and each sample in y.
    """
    return np.sqrt(np.sum((x - y) ** 2, axis=1))


class KNNClassifier(Model):
    """
    K-Nearest Neighbors classifier.
    Estimates the class for a sample based on the k most similar examples (shortest distance).

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

    def _fit(self, dataset: Dataset) -> 'KNNClassifier':
        """
        Stores the training dataset.

        Parameters
        ----------
        dataset: Dataset
            The training dataset.

        Returns
        -------
        self: KNNClassifier
        """
        self.dataset = dataset
        return self

    def _get_closest_label(self, sample: np.ndarray):
        """
        Returns the most common class among the k nearest examples to the given sample.

        Parameters
        ----------
        sample: numpy.ndarray (n_features,)
            A single sample.

        Returns
        -------
        label
            The estimated class for the sample.
        """
        distances = self.distance(sample, self.dataset.X)

        k_nearest_idxs = np.argsort(distances)[:self.k]
        k_nearest_labels = self.dataset.y[k_nearest_idxs]

        labels, counts = np.unique(k_nearest_labels, return_counts=True)
        return labels[np.argmax(counts)]

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Estimates the class for each sample in the testing dataset, based on
        the k most similar examples in the training dataset.

        Parameters
        ----------
        dataset: Dataset
            The testing dataset.

        Returns
        -------
        predictions: numpy.ndarray (n_samples,)
            An array of predicted classes for the testing dataset.
        """
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)

    def _score(self, dataset: Dataset) -> float:
        """
        Calculates the accuracy between the estimated classes and the actual ones.

        Parameters
        ----------
        dataset: Dataset
            The testing dataset.

        Returns
        -------
        error: float
            The accuracy between predictions and actual values.
        """
        predictions = self._predict(dataset)
        return accuracy(dataset.y, predictions)

import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset


class VarianceThreshold(Transformer):
    """
    VarianceThreshold is a feature selection method that removes all features whose
    variance does not meet a given threshold.

    Parameters
    ----------
    threshold: float
        Cutoff/cut-off value for the variance. Only features with a variance
        greater than this threshold are kept.

    Estimated Parameters
    ---------------------
    variance: numpy.ndarray
        The variance of each feature, estimated from the training data.
    """

    def __init__(self, threshold: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        if threshold < 0:
            raise ValueError("threshold must be a non-negative value")
        self.threshold = threshold
        self.variance = None

    def _fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """
        Estimates the variance of each feature.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit.

        Returns
        -------
        self: VarianceThreshold
        """
        self.variance = np.var(dataset.X, axis=0)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects all features with a variance greater than the threshold.

        Parameters
        ----------
        dataset: Dataset
            The dataset to transform.

        Returns
        -------
        dataset: Dataset
            The transformed Dataset object, with only the selected features.
        """
        mask = self.variance > self.threshold
        new_X = dataset.X[:, mask]
        new_features = np.array(dataset.features)[mask].tolist()

        return Dataset(new_X, dataset.y, features=new_features, label=dataset.label)

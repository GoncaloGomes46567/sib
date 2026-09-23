from typing import Callable

import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectKBest(Transformer):
    """
    SelectKBest selects features according to the k highest scores, computed with a
    scoring function such as f_classification.

    Parameters
    ----------
    score_func: Callable
        Variance analysis function that takes a Dataset object and returns a
        tuple (F, p) (f_classification by default).
    k: int
        Number of top features to select.

    Estimated Parameters
    ---------------------
    F: numpy.ndarray
        The F value for each feature, estimated by score_func.
    p: numpy.ndarray
        The p value for each feature, estimated by score_func.
    """

    def __init__(self, score_func: Callable = f_classification, k: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectKBest':
        """
        Estimates the F and p values for each feature using score_func.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit.

        Returns
        -------
        self: SelectKBest
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects the top k features with the highest F value.

        Parameters
        ----------
        dataset: Dataset
            The dataset to transform.

        Returns
        -------
        dataset: Dataset
            The transformed Dataset object, with only the selected features.
        """
        idxs = np.argsort(self.F)[-self.k:]
        idxs = np.sort(idxs)

        new_X = dataset.X[:, idxs]
        new_features = np.array(dataset.features)[idxs].tolist()

        return Dataset(new_X, dataset.y, features=new_features, label=dataset.label)

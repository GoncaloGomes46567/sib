from typing import Callable

import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile(Transformer):
    """
    SelectPercentile selects a given percentage of features based on their F-values,
    computed with a scoring function such as f_classification.

    Parameters
    ----------
    score_func: Callable
        Variance analysis function that takes a Dataset object and returns a
        tuple (F, p) (f_classification by default).
    percentile: float
        Percentile of features to select (e.g., 40 for the top 40% of features).

    Estimated Parameters
    ---------------------
    F: numpy.ndarray
        The F value for each feature, estimated by score_func.
    p: numpy.ndarray
        The p value for each feature, estimated by score_func.
    """

    def __init__(self, score_func: Callable = f_classification, percentile: float = 10, **kwargs):
        super().__init__(**kwargs)
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Estimates the F and p values for each feature using score_func.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit.

        Returns
        -------
        self: SelectPercentile
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects the top `percentile`% of features according to their F-value.
        Ties at the selection threshold are handled so that the exact number of
        features implied by the percentile is kept.

        Parameters
        ----------
        dataset: Dataset
            The dataset to transform.

        Returns
        -------
        dataset: Dataset
            The transformed Dataset object, with only the selected features.
        """
        n_features = len(self.F)
        n_select = int(round(n_features * self.percentile / 100))

        if n_select <= 0:
            new_X = dataset.X[:, :0]
            return Dataset(new_X, dataset.y, features=[], label=dataset.label)

        if n_select >= n_features:
            return Dataset(dataset.X.copy(), dataset.y, features=list(dataset.features), label=dataset.label)

        # threshold = F-value at the (100 - percentile)-th percentile
        threshold = np.percentile(self.F, 100 - self.percentile)

        # features strictly greater than the threshold are always selected
        mask = self.F > threshold
        n_selected = int(np.sum(mask))

        # fill the remaining slots with the first `remaining` features tied at the threshold
        remaining = n_select - n_selected
        if remaining > 0:
            tie_idxs = np.where(self.F == threshold)[0]
            mask[tie_idxs[:remaining]] = True

        idxs = np.where(mask)[0]
        idxs = np.sort(idxs)

        new_X = dataset.X[:, idxs]
        new_features = np.array(dataset.features)[idxs].tolist()

        return Dataset(new_X, dataset.y, features=new_features, label=dataset.label)

from typing import Tuple

import numpy as np
from scipy import stats

from si.data.dataset import Dataset


def f_classification(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scores features using a one-way ANOVA F-test between each feature and the class labels.

    Parameters
    ----------
    dataset: Dataset
        The Dataset object

    Returns
    -------
    F: numpy.ndarray
        F scores for each feature.
    p: numpy.ndarray
        p-values for each feature.
    """
    classes = dataset.get_classes()

    groups = [dataset.X[dataset.y == c, :] for c in classes]

    F, p = stats.f_oneway(*groups)

    return F, p

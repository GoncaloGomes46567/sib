from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = None) -> Tuple[Dataset, Dataset]:
    """
    Splits a Dataset object into training and testing Dataset objects.

    Parameters
    ----------
    dataset: Dataset
        The Dataset object to split into training and testing data
    test_size: float
        The size of the testing Dataset (e.g., 0.2 for 20%)
    random_state: int
        Seed for generating permutations

    Returns
    -------
    train, test: Tuple[Dataset, Dataset]
        A tuple containing the train and test Dataset objects.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = dataset.shape()[0]
    permutations = np.random.permutation(n_samples)

    n_test = int(n_samples * test_size)

    test_idxs = permutations[:n_test]
    train_idxs = permutations[n_test:]

    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs] if dataset.y is not None else None,
                     features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs] if dataset.y is not None else None,
                    features=dataset.features, label=dataset.label)

    return train, test


def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2,
                                 random_state: int = None) -> Tuple[Dataset, Dataset]:
    """
    Splits a Dataset object into stratified training and testing Dataset objects,
    keeping (approximately) the same class proportions in both sets.

    Parameters
    ----------
    dataset: Dataset
        The Dataset object to split into training and testing data
    test_size: float
        The size of the testing Dataset (e.g., 0.2 for 20%)
    random_state: int
        Seed for generating permutations

    Returns
    -------
    train, test: Tuple[Dataset, Dataset]
        A tuple containing the stratified train and test Dataset objects.
    """
    if random_state is not None:
        np.random.seed(random_state)

    labels, counts = np.unique(dataset.y, return_counts=True)

    train_idxs = []
    test_idxs = []

    for label, count in zip(labels, counts):
        label_idxs = np.where(dataset.y == label)[0]

        n_test = int(count * test_size)

        shuffled = np.random.permutation(label_idxs)
        test_idxs.extend(shuffled[:n_test])
        train_idxs.extend(shuffled[n_test:])

    train_idxs = np.array(train_idxs)
    test_idxs = np.array(test_idxs)

    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)

    return train, test

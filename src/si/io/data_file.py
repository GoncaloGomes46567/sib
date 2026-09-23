import numpy as np

from si.data.dataset import Dataset


def read_data_file(filename: str, sep: str = ",", label: bool = False) -> Dataset:
    data = np.genfromtxt(filename, delimiter=sep)

    if label:
        X = data[:, :-1]
        y = data[:, -1]
    else:
        X = data
        y = None

    return Dataset(X, y)


def write_data_file(filename: str, dataset: Dataset, sep: str = ",", label: bool = False) -> None:
 
    if label and dataset.has_label():
        y = dataset.y.reshape(-1, 1)
        data = np.hstack((dataset.X, y))
    else:
        data = dataset.X

    np.savetxt(filename, data, delimiter=sep)

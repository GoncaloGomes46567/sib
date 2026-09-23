import pandas as pd

from si.data.dataset import Dataset


def read_csv(filename: str, sep: str = ",", features: bool = False, label: bool = False) -> Dataset:
    """
    Reads a csv file and returns a Dataset object.
    The file is always assumed to have a header row (with column names); the
    'features' argument controls whether those names are used as the dataset's
    feature names or discarded (using generic names instead).

    Parameters
    ----------
    filename: str
        Name/path of the file
    sep: str
        The value separator
    features: bool
        Whether to use the file's header row as feature names
    label: bool
        Whether the file has y (assumed to be the last column)

    Returns
    -------
    Dataset
    """
    df = pd.read_csv(filename, sep=sep, header=0)

    if label:
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        feature_names = list(df.columns[:-1]) if features else None
        label_name = df.columns[-1] if features else None
    else:
        X = df.to_numpy()
        y = None
        feature_names = list(df.columns) if features else None
        label_name = None

    return Dataset(X, y, features=feature_names, label=label_name)


def write_csv(filename: str, dataset: Dataset, sep: str = ",", features: bool = False, label: bool = False) -> None:
    """
    Writes a Dataset object to a csv file. A header row is always written so
    that the file can be read back with read_csv.

    Parameters
    ----------
    filename: str
        Name/path of the file
    dataset: Dataset
        Dataset object to write to the file
    sep: str
        The value separator
    features: bool
        Whether to use the dataset's actual feature names in the header
        (otherwise generic names are used)
    label: bool
        Whether to write y
    """
    if features and dataset.features is not None:
        columns = list(dataset.features)
    else:
        columns = [f"feat_{i}" for i in range(dataset.X.shape[1])]

    data = pd.DataFrame(dataset.X, columns=columns)

    if label and dataset.has_label():
        label_name = dataset.label if (features and dataset.label is not None) else "y"
        data[label_name] = dataset.y

    data.to_csv(filename, sep=sep, index=False, header=True)

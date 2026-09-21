import pandas as pd


def read_csv(filename, sep=",", features=True, label=True):

    df = pd.read_csv(
        filename,
        sep=sep,
        header=0 if features else None
    )

    if features:
        feature_names = list(df.columns)
    else:
        feature_names = None

    if label:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    else:
        X = df.values
        y = None

    return X, y, feature_names


def write_csv(filename, dataset, sep=",", features=True, label=True):

    data = dataset.X

    if label:
        data = pd.DataFrame(data)
        data["y"] = dataset.y
    else:
        data = pd.DataFrame(data)

    if features:
        if hasattr(dataset, "feature_names"):
            columns = list(dataset.feature_names)

            if label:
                columns.append("y")

            data.columns = columns

    data.to_csv(
        filename,
        sep=sep,
        index=False,
        header=features
    )
from typing import Tuple, Sequence, Union

import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None) -> None:
     
        if X is None:
            raise ValueError("X cannot be None")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if features is not None and len(X[0]) != len(features):
            raise ValueError("Number of features must match the number of columns in X")
        if features is None:
            features = [f"feat_{str(i)}" for i in range(X.shape[1])]
        if y is not None and label is None:
            label = "y"
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> Tuple[int, int]:
      
        return self.X.shape

    def has_label(self) -> bool:
     
        return self.y is not None

    def get_classes(self) -> np.ndarray:
     
        if self.has_label():
            return np.unique(self.y)
        else:
            raise ValueError("Dataset does not have a label")

    def get_mean(self) -> np.ndarray:
        return np.nanmean(self.X, axis=0)

    def get_variance(self) -> np.ndarray:
        return np.nanvar(self.X, axis=0)

    def get_median(self) -> np.ndarray:
        return np.nanmedian(self.X, axis=0)

    def get_min(self) -> np.ndarray:
        return np.nanmin(self.X, axis=0)

    def get_max(self) -> np.ndarray:
        return np.nanmax(self.X, axis=0)

    def summary(self) -> pd.DataFrame:
        data = {
            "mean": self.get_mean(),
            "median": self.get_median(),
            "min": self.get_min(),
            "max": self.get_max(),
            "var": self.get_variance()
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=self.features)

    def dropna(self) -> 'Dataset':
        mask = ~np.isnan(self.X).any(axis=1)
        self.X = self.X[mask]
        if self.y is not None:
            self.y = self.y[mask]
        return self

    def fillna(self, value: Union[float, str]) -> 'Dataset':
        if value == "mean":
            fill_values = self.get_mean()
        elif value == "median":
            fill_values = self.get_median()
        elif isinstance(value, (int, float)):
            fill_values = np.full(self.X.shape[1], value)
        else:
            raise ValueError('value must be a float, "mean" or "median"')

        nan_mask = np.isnan(self.X)
        col_idxs = np.where(nan_mask)[1]
        self.X[nan_mask] = fill_values[col_idxs]
        return self

    def remove_by_index(self, index: int) -> 'Dataset':
        self.X = np.delete(self.X, index, axis=0)
        if self.y is not None:
            self.y = np.delete(self.y, index, axis=0)
        return self

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, label: str = None):
        if label:
            X = df.drop(label, axis=1).to_numpy()
            y = df[label].to_numpy()
        else:
            X = df.to_numpy()
            y = None

        features = df.columns.tolist()
        return cls(X, y, features=features, label=label)

    def to_dataframe(self) -> pd.DataFrame:
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df

    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)


if __name__ == '__main__':
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2])
    features = np.array(['a', 'b', 'c'])
    label = 'y'
    dataset = Dataset(X, y, features, label)
    print(dataset.shape())
    print(dataset.has_label())
    print(dataset.get_classes())
    print(dataset.get_mean())
    print(dataset.get_variance())
    print(dataset.get_median())
    print(dataset.get_min())
    print(dataset.get_max())
    print(dataset.summary())

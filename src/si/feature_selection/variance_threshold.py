import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset


class VarianceThreshold(Transformer):
 

    def __init__(self, threshold: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        if threshold < 0:
            raise ValueError("threshold must be a non-negative value")
        self.threshold = threshold
        self.variance = None

    def _fit(self, dataset: Dataset) -> 'VarianceThreshold':
      
        self.variance = np.var(dataset.X, axis=0)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
    
        mask = self.variance > self.threshold
        new_X = dataset.X[:, mask]
        new_features = np.array(dataset.features)[mask].tolist()

        return Dataset(new_X, dataset.y, features=new_features, label=dataset.label)

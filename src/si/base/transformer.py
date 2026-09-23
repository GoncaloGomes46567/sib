from abc import abstractmethod

from si.base.estimator import Estimator
from si.data.dataset import Dataset


class Transformer(Estimator):


    def transform(self, dataset: Dataset) -> Dataset:
       
        if not self.is_fitted():
            raise ValueError('Transformer needs to be fitted before calling transform()')
        return self._transform(dataset)

    @abstractmethod
    def _transform(self, dataset: Dataset) -> Dataset:
  

    def fit_transform(self, dataset: Dataset) -> Dataset:
    
        return self.fit(dataset).transform(dataset)

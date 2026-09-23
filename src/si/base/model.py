from abc import ABCMeta, ABC, abstractmethod

from si.base.estimator import Estimator


class Model(Estimator, ABC):

    def __init__(self, **kwargs):
 
        super().__init__(**kwargs)

    def predict(self, dataset):
       
        if not self.is_fitted():
            raise ValueError('Model needs to be fitted before calling predict()')
        return self._predict(dataset)

    @abstractmethod
    def _predict(self, dataset):
 
    def fit_predict(self, dataset):
       
        self.fit(dataset)
        return self.predict(dataset)

    @abstractmethod
    def _score(self, dataset: 'Dataset') -> float:
    

    def score(self, dataset) -> float:
     
        if not self.is_fitted():
            raise ValueError('Model needs to be fitted before calling score()')
        return self._score(dataset)

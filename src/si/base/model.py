from abc import ABCMeta, ABC, abstractmethod

from si.base.estimator import Estimator


class Model(Estimator, ABC):
    """
    Abstract base class for models.
    A model is an object that can predict the target values of a Dataset object.
    """

    def __init__(self, **kwargs):
        """
        Initialize the model.
        """
        super().__init__(**kwargs)

    def predict(self, dataset):
        """
        Predict the target values of the dataset.
        The model needs to be fitted before calling this method.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the target values of.

        Returns
        -------
        predictions: np.ndarray
            The predicted target values.
        """
        if not self.is_fitted():
            raise ValueError('Model needs to be fitted before calling predict()')
        return self._predict(dataset)

    @abstractmethod
    def _predict(self, dataset):
        """
        Predict the target values of the dataset.
        Abstract method that needs to be implemented by all subclasses.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the target values of.

        Returns
        -------
        predictions: np.ndarray
            The predicted target values.
        """

    def fit_predict(self, dataset):
        """
        Fit the model to the dataset and predict the target values.
        Equivalent to calling fit(dataset) and then predict(dataset).

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit and predict the target values of.

        Returns
        -------
        predictions: np.ndarray
            The predicted target values.
        """
        self.fit(dataset)
        return self.predict(dataset)

    @abstractmethod
    def _score(self, dataset: 'Dataset') -> float:
        """
        Calculate the error between the estimated values and the actual values (dataset.y).
        Abstract method that needs to be implemented by all subclasses.
        Typically implemented by getting the predictions (via _predict) and then
        comparing them to dataset.y using the appropriate error metric.

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on.

        Returns
        -------
        score: float
            The error/score metric.
        """

    def score(self, dataset) -> float:
        """
        Verifies whether the model is fitted and, if so, calls _score to compute
        the error/score metric on the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on.

        Returns
        -------
        score: float
            The error/score metric.
        """
        if not self.is_fitted():
            raise ValueError('Model needs to be fitted before calling score()')
        return self._score(dataset)

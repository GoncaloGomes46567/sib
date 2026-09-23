import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the accuracy between the real and predicted labels.

    Parameters
    ----------
    y_true: numpy.ndarray
        The real label values
    y_pred: numpy.ndarray
        The predicted label values

    Returns
    -------
    accuracy: float
        The portion of well classified samples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sum(y_true == y_pred) / len(y_true)

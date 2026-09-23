import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
 
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true))

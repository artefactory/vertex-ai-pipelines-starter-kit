import numpy as np
from sklearn.metrics import mean_squared_error


def forecast_accuracy(y_true: np.array, y_pred: np.array):
    """Calculates the forecast accuracy (= 1 - MAPE).

    Parameters
    ----------
    y_true : np.array
        True values.
    y_pred : np.array
        Forecasted values.

    Returns
    -------
    float
        Forecast accuracy (between 0 and 1).
    """
    try:
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        mask = y_true != 0
        fa_vect = 1 - np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        fa_vect_clipped = np.clip(fa_vect, a_min=0, a_max=1)
        return np.mean(fa_vect_clipped)
    except Exception as e:
        print(e)
        return 0


def evaluate(y_true, y_pred, name_set):

    FA = forecast_accuracy(y_true, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"Performance on {name_set} set:")
    print(f"FA = {FA}")
    print(f"RMSE = {RMSE}")

    return FA, RMSE

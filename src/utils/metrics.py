import numpy as np
from scipy.stats import spearmanr


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error. Range [0, 200]."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = (np.abs(actual) + np.abs(predicted)) / 2.0
    mask = denom != 0
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)


def nfa(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Normalized Forecasting Accuracy = 1 - SMAPE/200. Range [0, 1]."""
    return max(0.0, 1.0 - smape(actual, predicted) / 200.0)


def spearman_corr(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Spearman Rank Correlation Coefficient."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) < 3:
        return float("nan")
    corr, _ = spearmanr(actual, predicted)
    return float(corr)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def compute_all_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute all metrics and return as dict."""
    return {
        "NFA": nfa(actual, predicted),
        "SC": spearman_corr(actual, predicted),
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "SMAPE": smape(actual, predicted),
    }

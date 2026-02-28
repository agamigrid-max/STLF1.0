from typing import Dict
import numpy as np


def mbe(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(y_pred - y_true))


def pbias(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.sum(y_true)
    if denom == 0:
        return float("nan")
    return float(100.0 * np.sum(y_pred - y_true) / denom)


def error_variance(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    errors = y_pred - y_true
    return float(np.var(errors))


def compute_reliability_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "mbe": mbe(y_true, y_pred),
        "pbias": pbias(y_true, y_pred),
        "error_variance": error_variance(y_true, y_pred),
    }

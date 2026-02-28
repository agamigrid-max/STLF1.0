from typing import Dict
import numpy as np
from sklearn.metrics import r2_score


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def md_ae(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.median(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float("nan")


def nrmse(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.max(y_true) - np.min(y_true)
    if denom == 0:
        return float("nan")
    return rmse(y_true, y_pred) / float(denom)


def wape(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def compute_accuracy_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "md_ae": md_ae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred),
        "wape": wape(y_true, y_pred),
    }

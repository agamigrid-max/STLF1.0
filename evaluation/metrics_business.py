from typing import Dict, Optional, Sequence
import numpy as np
from .metrics_accuracy import mae

def peak_demand_mae(y_true, y_pred, peak_mask=None):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if peak_mask is None:
        threshold = np.quantile(y_true, 0.9)
        peak_mask = y_true >= threshold
    else:
        peak_mask = np.array(peak_mask)
    return mae(y_true[peak_mask], y_pred[peak_mask]) if np.any(peak_mask) else float("nan")

def energy_weighted_mape(y_true, y_pred, energy_weights=None):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if energy_weights is None: energy_weights = y_true
    energy_weights = np.array(energy_weights)
    mask = (y_true != 0) & (energy_weights > 0)
    if not np.any(mask): return float("nan")
    abs_pct = np.abs((y_true[mask]-y_pred[mask]) / y_true[mask])
    return float(100*np.sum(abs_pct * energy_weights[mask]) / np.sum(energy_weights[mask]))

def compute_business_metrics(y_true, y_pred, peak_mask=None, energy_weights=None) -> Dict[str, float]:
    return {
        "peak_demand_mae": peak_demand_mae(y_true, y_pred, peak_mask),
        "energy_weighted_mape": energy_weighted_mape(y_true, y_pred, energy_weights),
    }

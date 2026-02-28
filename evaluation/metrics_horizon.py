from typing import Dict, Sequence
import numpy as np
from .metrics_accuracy import mae

def compute_mae_per_horizon(y_true, y_pred, horizons, max_horizon=24) -> Dict[str, float]:
    y_true, y_pred, horizons = map(np.array, (y_true, y_pred, horizons))
    result = {}
    for h in range(1, max_horizon+1):
        mask = horizons == h
        result[str(h)] = mae(y_true[mask], y_pred[mask]) if np.any(mask) else float("nan")
    return result

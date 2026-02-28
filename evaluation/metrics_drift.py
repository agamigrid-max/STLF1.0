from typing import Dict, Optional

def compute_prediction_drift_status(current_mape, baseline_mape=None, warning_ratio=1.2, drift_ratio=1.5):
    if baseline_mape is None or baseline_mape != baseline_mape: return "unknown"
    if current_mape <= baseline_mape * warning_ratio: return "stable"
    if current_mape <= baseline_mape * drift_ratio: return "warning"
    return "drifting"

def compute_drift_metrics(current_mape, baseline_mape=None) -> Dict[str, str]:
    return {"prediction_drift_status": compute_prediction_drift_status(current_mape, baseline_mape)}

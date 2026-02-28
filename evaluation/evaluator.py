from typing import Any, Dict, Optional, Sequence
from .metrics_accuracy import compute_accuracy_metrics
from .metrics_reliability import compute_reliability_metrics
from .metrics_efficiency import compute_efficiency_metrics
from .metrics_horizon import compute_mae_per_horizon
from .metrics_drift import compute_drift_metrics
from .metrics_business import compute_business_metrics

def evaluate_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    horizons: Optional[Sequence[int]] = None,
    runtime_stats: Optional[Dict[str, Any]] = None,
    baseline_mape: Optional[float] = None,
    peak_mask: Optional[Sequence[bool]] = None,
    energy_weights: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:

    accuracy = compute_accuracy_metrics(y_true, y_pred)
    reliability = compute_reliability_metrics(y_true, y_pred)
    efficiency = compute_efficiency_metrics(runtime_stats)

    horizon = {
        "mae_per_horizon": compute_mae_per_horizon(y_true, y_pred, horizons)
    } if horizons is not None else {"mae_per_horizon": {}}

    drift = compute_drift_metrics(
        current_mape=accuracy.get("mape", float("nan")),
        baseline_mape=baseline_mape,
    )

    business = compute_business_metrics(
        y_true, y_pred, peak_mask, energy_weights
    )

    return {
        "accuracy": accuracy,
        "reliability": reliability,
        "efficiency": efficiency,
        "horizon": horizon,
        "drift": drift,
        "business": business,
    }

from typing import Dict, Optional

def compute_efficiency_metrics(runtime_stats: Optional[dict] = None) -> Dict[str, float]:
    runtime_stats = runtime_stats or {}
    return {
        "training_time_sec": float(runtime_stats.get("training_time_sec", float("nan"))),
        "inference_time_sec": float(runtime_stats.get("inference_time_sec", float("nan"))),
        "model_size_mb": float(runtime_stats.get("model_size_mb", float("nan"))),
    }

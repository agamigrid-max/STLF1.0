from evaluation import evaluate_metrics
import numpy as np

def main():
    # Simple synthetic test data
    y_true = np.array([100, 120, 130, 110, 90, 80])
    y_pred = np.array([98, 125, 128, 115, 85, 82])
    horizons = [1, 2, 3, 4, 5, 6]

    runtime_stats = {
        "training_time_sec": 12.3,
        "inference_time_sec": 0.045,
        "model_size_mb": 3.2,
    }

    metrics = evaluate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        horizons=horizons,
        runtime_stats=runtime_stats,
        baseline_mape=5.0,
    )

    from pprint import pprint
    pprint(metrics)

if __name__ == "__main__":
    main()

"""
End-to-end training pipeline for STLF.
"""

import pandas as pd
import pickle

from ingestion.load_data import load_csv
from preprocessing.clean_data import clean
from preprocessing.resample import enforce_hourly_frequency
from preprocessing.outliers import clip_outliers

from features.lags import add_lag_features
from features.rolling import add_rolling_features
from features.calendar import add_calendar_features

from models.baseline import (
    train_baseline_model,
    predict_baseline,
    mae,
    rmse
)


def run_pipeline(data_path="data/synthetic_load.csv",
                 model_path="models/baseline_model.pkl"):

    print("Loading data...")
    df = load_csv(data_path)

    print("Preprocessing...")
    df = clean(df)
    df = enforce_hourly_frequency(df)
    df = clip_outliers(df)

    print("Feature engineering...")
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)

    df = df.dropna().reset_index(drop=True)

    train_df = df.iloc[:-168]
    test_df = df.iloc[-168:]

    print("Training baseline model...")
    model, feature_cols = train_baseline_model(train_df)

    print("Generating predictions...")
    preds = predict_baseline(model, test_df, feature_cols)

    print("Evaluating...")
    mae_val = mae(test_df["load"], preds)
    rmse_val = rmse(test_df["load"], preds)

    print(f"MAE: {mae_val}")
    print(f"RMSE: {rmse_val}")

    with open(model_path, "wb") as f:
        pickle.dump((model, feature_cols), f)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    run_pipeline()
# pipeline/train.py

import time
import joblib
import numpy as np
from pathlib import Path

from evaluation import evaluate_metrics


def train_pipeline(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    model_save_path="models/baseline_model.pkl",
    baseline_mape=None,
    horizons=None,
    peak_mask=None,
    energy_weights=None,
):
    """
    Runs training, evaluates metrics, saves model, and returns metrics JSON.
    """

    # ----------------------------
    # 1. Training with timing
    # ----------------------------
    t0 = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - t0

    # ----------------------------
    # 2. Inference timing
    # ----------------------------
    t1 = time.time()
    y_pred = model.predict(X_val)
    inference_time = time.time() - t1

    # ----------------------------
    # 3. Model size
    # ----------------------------
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, model_save_path)
    model_size_mb = Path(model_save_path).stat().st_size / (1024 * 1024)

    # ----------------------------
    # 4. Runtime stats for efficiency metrics
    # ----------------------------
    runtime_stats = {
        "training_time_sec": training_time,
        "inference_time_sec": inference_time,
        "model_size_mb": model_size_mb,
    }

    # ----------------------------
    # 5. Evaluate metrics using the new engine
    # ----------------------------
    metrics = evaluate_metrics(
        y_true=np.array(y_val),
        y_pred=np.array(y_pred),
        horizons=horizons,
        runtime_stats=runtime_stats,
        baseline_mape=baseline_mape,
        peak_mask=peak_mask,
        energy_weights=energy_weights,
    )

    # ----------------------------
    # 6. Return everything needed by API/UI
    # ----------------------------
    return {
        "model_path": model_save_path,
        "metrics": metrics,
        "training_time_sec": training_time,
        "inference_time_sec": inference_time,
        "model_size_mb": model_size_mb,
    }

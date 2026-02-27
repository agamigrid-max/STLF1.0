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

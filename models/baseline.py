"""
Baseline model for STLF using Linear Regression.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

def train_baseline_model(df: pd.DataFrame, target_col="load") -> LinearRegression:
    """
    Trains a simple Linear Regression model using all feature columns
    except the target and timestamp.
    """
    df = df.dropna().copy()

    feature_cols = [c for c in df.columns if c not in ["timestamp", target_col]]
    X = df[feature_cols]
    y = df[target_col]

    model = LinearRegression()
    model.fit(X, y)

    return model, feature_cols

def predict_baseline(model: LinearRegression, df: pd.DataFrame, feature_cols) -> pd.Series:
    """
    Generates predictions using the trained baseline model.
    """
    X = df[feature_cols]
    preds = model.predict(X)
    return pd.Series(preds, index=df.index)

"""
Lag feature generation for STLF.
"""

import pandas as pd

def add_lag_features(df: pd.DataFrame, lags=[1, 24, 168]) -> pd.DataFrame:
    """
    Adds lag features for the load column.
    Default lags:
    - 1 hour
    - 24 hours (same hour previous day)
    - 168 hours (same hour previous week)
    """
    df = df.copy()
    for lag in lags:
        df[f"load_lag_{lag}"] = df["load"].shift(lag)
    return df

"""
Rolling window feature generation.
"""

import pandas as pd

def add_rolling_features(df: pd.DataFrame, windows=[3, 24, 168]) -> pd.DataFrame:
    """
    Adds rolling mean and std for the load column.
    Default windows:
    - 3 hours (short-term smoothing)
    - 24 hours (daily pattern)
    - 168 hours (weekly pattern)
    """
    df = df.copy()
    for w in windows:
        df[f"load_roll_mean_{w}"] = df["load"].rolling(w).mean()
        df[f"load_roll_std_{w}"] = df["load"].rolling(w).std()
    return df

"""
Calendar/time-based feature generation.
"""

import pandas as pd

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds calendar features derived from the timestamp column.
    """
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    return df

"""
Resampling utilities for STLF preprocessing.
"""

import pandas as pd

def enforce_hourly_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures the DataFrame has a continuous hourly frequency.
    Missing timestamps are filled, and load values are forward-filled.
    """
    df = df.copy()
    df = df.set_index("timestamp")

    # Create a full hourly index (lowercase 'h' is the new standard)
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")

    # Reindex and fill missing values
    df = df.reindex(full_index)
    df["load"] = df["load"].ffill()

    df = df.reset_index().rename(columns={"index": "timestamp"})
    return df

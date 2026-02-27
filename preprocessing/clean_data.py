"""
Basic cleaning utilities for STLF preprocessing.
"""

import pandas as pd

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DataFrame by:
    - Dropping rows with missing timestamps
    - Forward-filling missing load values
    """
    df = df.copy()

    # Drop rows with missing timestamps
    df = df.dropna(subset=["timestamp"])

    # Forward-fill missing load values
    if df["load"].isna().any():
        df["load"] = df["load"].ffill()

    return df

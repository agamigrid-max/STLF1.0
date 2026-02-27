"""
Utility functions for ingestion module.
"""

import pandas as pd

def parse_timestamp_column(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    """
    Ensures the timestamp column is converted to pandas datetime.
    """
    if column not in df.columns:
        raise ValueError(f"Timestamp column '{column}' not found in DataFrame.")

    df[column] = pd.to_datetime(df[column], errors="coerce")

    if df[column].isna().any():
        raise ValueError("Some timestamp values could not be parsed.")

    return df

"""
CSV ingestion and schema validation for STLF pipeline.
"""

import pandas as pd
from .schema import REQUIRED_COLUMNS, OPTIONAL_COLUMNS
from .utils import parse_timestamp_column


def validate_schema(df: pd.DataFrame):
    """
    Ensures required columns exist in the DataFrame.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_csv(path: str) -> pd.DataFrame:
    """
    Loads a CSV file, validates schema, parses timestamps,
    and returns a clean DataFrame.
    """
    df = pd.read_csv(path)

    # Validate required columns
    validate_schema(df)

    # Parse timestamp column
    df = parse_timestamp_column(df, "timestamp")

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df

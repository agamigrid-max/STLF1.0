"""
Schema definitions for the ingestion module.
Defines required and optional columns for the STLF pipeline input.
"""

from typing import List

# Core time series structure
REQUIRED_COLUMNS: List[str] = [
    "timestamp",  # datetime-like, parsed to pandas.DatetimeIndex
    "load",       # target variable for STLF (MW)
]

# Optional exogenous features
OPTIONAL_COLUMNS: List[str] = [
    "temperature",
    "humidity",
    "wind_speed",
    "day_of_week",
    "is_holiday",
]

ALL_KNOWN_COLUMNS: List[str] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

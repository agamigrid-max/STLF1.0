"""
Feature engineering module for STLF.

Includes:
- Lag features
- Rolling window statistics
- Calendar/time-based features
"""

from .lags import add_lag_features
from .rolling import add_rolling_features
from .calendar import add_calendar_features

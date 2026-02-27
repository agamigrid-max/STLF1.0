"""
Outlier detection and correction utilities.
"""

import pandas as pd
import numpy as np

def detect_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.Series:
    """
    Returns a boolean Series indicating which load values are outliers
    based on z-score threshold.
    """
    load = df["load"]
    z_scores = (load - load.mean()) / load.std()
    return z_scores.abs() > z_thresh

def clip_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Clips outliers to the z-threshold range.
    """
    df = df.copy()
    outliers = detect_outliers(df, z_thresh)

    mean = df["load"].mean()
    std = df["load"].std()
    upper = mean + z_thresh * std
    lower = mean - z_thresh * std

    df.loc[outliers, "load"] = np.clip(df.loc[outliers, "load"], lower, upper)
    return df

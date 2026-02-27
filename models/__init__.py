"""
Modeling package for STLF.

Includes:
- Baseline models
- Evaluation metrics
"""

from .baseline import train_baseline_model, predict_baseline
from .metrics import mae, rmse

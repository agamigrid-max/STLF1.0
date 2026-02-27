"""
Ingestion package for the STLF project.

Responsible for:
- Loading raw data from CSV/DB/API sources.
- Validating and standardizing the input schema.
- Returning a clean pandas DataFrame to downstream modules.
"""

from .load_data import load_csv
from .schema import REQUIRED_COLUMNS, OPTIONAL_COLUMNS

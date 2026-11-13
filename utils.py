"""
AgriPulse - General Utility & Helper Functions
Provides file loading, number formatting, data conversion, and export helpers.
"""

import os
import io
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def format_large_number(num: float, unit: str = '') -> str:
    """Formats large agricultural figures with readable suffixes (K, M, B)."""
    if pd.isna(num):
        return "N/A"
    abs_num = abs(num)
    if abs_num >= 1e9:
        val = f"{num / 1e9:.2f} B"
    elif abs_num >= 1e6:
        val = f"{num / 1e6:.2f} M"
    elif abs_num >= 1e3:
        val = f"{num / 1e3:.1f} K"
    else:
        val = f"{num:.2f}"
    return f"{val} {unit}".strip()


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Converts a pandas DataFrame into a downloadable UTF-8 CSV byte stream."""
    return df.to_csv(index=False).encode('utf-8')


def load_dataset(file_or_path, is_uploaded: bool = False) -> pd.DataFrame:
    """Loads a CSV or Excel file safely into a pandas DataFrame."""
    if is_uploaded:
        filename = file_or_path.name
        if filename.endswith('.csv'):
            return pd.read_csv(file_or_path)
        elif filename.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file_or_path)
        else:
            raise ValueError("Unsupported file format. Please upload a .csv or .xlsx file.")
    else:
        if os.path.exists(file_or_path):
            return pd.read_csv(file_or_path)
        else:
            raise FileNotFoundError(f"File not found: {file_or_path}")

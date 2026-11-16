"""Nonmutating missing-value masks shared by overview profilers."""

from __future__ import annotations

import pandas as pd


def blank_mask(series: pd.Series) -> pd.Series:
    """Mark empty or whitespace-only strings, excluding pandas null values."""
    return series.astype("string").str.fullmatch(r"\s*", na=False).astype(bool)


def missing_mask(series: pd.Series) -> pd.Series:
    """Mark pandas nulls and blank strings without changing source values."""
    return series.isna() | blank_mask(series)

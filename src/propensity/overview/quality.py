"""Basic quality profiling for column missingness/blanks and exact duplicates."""

from __future__ import annotations

import pandas as pd

from propensity.overview.missing import blank_mask, missing_mask
from propensity.overview.models import (
    QUALITY_SUMMARY_CONTRACT,
    normalize_result_frame,
)
from propensity.schema.spec import DatasetSpec


def profile_quality(
    frame: pd.DataFrame,
    table_name: str,
    spec: DatasetSpec | None = None,
) -> pd.DataFrame:
    """Profile missingness, blank strings, constant columns, and full-row duplicates."""
    n_rows = len(frame)
    rows: list[dict[str, object]] = []

    # 1. Column-level quality metrics
    for col in frame.columns:
        b_mask = blank_mask(frame[col])
        m_mask = missing_mask(frame[col])
        n_blank = int(b_mask.sum())
        n_missing = int(m_mask.sum())
        missing_pct = float(n_missing / n_rows * 100.0) if n_rows > 0 else pd.NA

        non_m = frame.loc[~m_mask, col]
        n_unique_non_missing = int(non_m.nunique(dropna=False))
        is_all_missing = bool(n_missing == n_rows) if n_rows > 0 else False
        is_constant = (
            bool(n_unique_non_missing == 1 and not is_all_missing)
            if n_rows > 0
            else False
        )

        rows.append(
            {
                "table": table_name,
                "column": str(col),
                "n_rows": n_rows,
                "n_missing": n_missing,
                "n_blank": n_blank,
                "missing_pct": missing_pct,
                "n_unique_non_missing": n_unique_non_missing,
                "is_all_missing": is_all_missing,
                "is_constant": is_constant,
                "duplicate_groups": pd.NA,
                "excess_rows": pd.NA,
                "duplicate_pct": pd.NA,
            }
        )

    # 2. Table-level exact duplicates row
    if n_rows == 0 or len(frame.columns) == 0:
        duplicate_groups: object = 0
        excess_rows: object = 0
        duplicate_pct: object = pd.NA
    else:
        dups = frame.duplicated(keep=False)
        if dups.any():
            dup_groups = frame[dups].groupby(list(frame.columns), dropna=False).size()
            duplicate_groups = len(dup_groups)
            excess_rows = int((dup_groups - 1).sum())
            duplicate_pct = float(excess_rows / n_rows * 100.0)
        else:
            duplicate_groups = 0
            excess_rows = 0
            duplicate_pct = 0.0

    rows.append(
        {
            "table": table_name,
            "column": None,
            "n_rows": n_rows,
            "n_missing": pd.NA,
            "n_blank": pd.NA,
            "missing_pct": pd.NA,
            "n_unique_non_missing": pd.NA,
            "is_all_missing": pd.NA,
            "is_constant": pd.NA,
            "duplicate_groups": duplicate_groups,
            "excess_rows": excess_rows,
            "duplicate_pct": duplicate_pct,
        }
    )

    result = pd.DataFrame(rows)
    return normalize_result_frame(result, QUALITY_SUMMARY_CONTRACT)

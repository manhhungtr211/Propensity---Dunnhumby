"""Numeric distribution profiling similar to df.describe() for numeric columns."""

from __future__ import annotations

import pandas as pd
from pandas.api import types as ptypes

from propensity.overview.missing import missing_mask
from propensity.overview.models import (
    NUMERIC_SUMMARY_CONTRACT,
    normalize_result_frame,
)
from propensity.schema.spec import DatasetSpec


def profile_numeric(
    frame: pd.DataFrame,
    table_name: str,
    spec: DatasetSpec | None = None,
) -> pd.DataFrame:
    """Profile distribution of numeric columns: count, mean, std, min, p25, median, p75, max.

    Filters for columns whose semantic role is ColumnRole.NUMERIC (if spec is provided),
    or physical numeric columns (excluding boolean).
    """
    if len(frame.columns) == 0:
        return normalize_result_frame(
            pd.DataFrame(columns=list(NUMERIC_SUMMARY_CONTRACT.columns)),
            NUMERIC_SUMMARY_CONTRACT,
        )

    numeric_cols: list[str] = []
    for col in frame.columns:
        dtype = frame.dtypes[col]
        series = frame[col]

        if spec is not None and (
            col in spec.identifier_cols
            or col in spec.datetime_cols
            or col in spec.ordinal_time_cols
            or col in spec.clock_time_cols
            or col == spec.time_col
            or any(col in endpoints for endpoints in spec.interval_cols)
            or col == spec.entity_col
        ):
            continue

        if ptypes.is_bool_dtype(dtype):
            continue

        if ptypes.is_numeric_dtype(dtype):
            numeric_cols.append(col)
        elif dtype == "object":
            # If object, check if non-blank values are numeric
            m = missing_mask(series)
            s_clean = series.loc[~m]
            if (
                len(s_clean) > 0
                and pd.to_numeric(s_clean.head(10), errors="coerce").notna().all()
            ):
                numeric_cols.append(col)

    if not numeric_cols:
        return normalize_result_frame(
            pd.DataFrame(columns=list(NUMERIC_SUMMARY_CONTRACT.columns)),
            NUMERIC_SUMMARY_CONTRACT,
        )

    rows: list[dict[str, object]] = []
    for col in numeric_cols:
        series = frame[col]
        m = missing_mask(series)
        valid = series.loc[~m]
        valid_numeric = pd.to_numeric(valid, errors="coerce").dropna()

        count = len(valid_numeric)
        if count == 0:
            mean_val: object = pd.NA
            std_val: object = pd.NA
            min_val: object = pd.NA
            p25_val: object = pd.NA
            med_val: object = pd.NA
            p75_val: object = pd.NA
            max_val: object = pd.NA
        else:
            mean_val = float(valid_numeric.mean())
            std_val = float(valid_numeric.std(ddof=1)) if count > 1 else pd.NA
            min_val = float(valid_numeric.min())
            p25_val = float(valid_numeric.quantile(0.25))
            med_val = float(valid_numeric.median())
            p75_val = float(valid_numeric.quantile(0.75))
            max_val = float(valid_numeric.max())

        rows.append(
            {
                "table": table_name,
                "column": str(col),
                "count": count,
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "p25": p25_val,
                "median": med_val,
                "p75": p75_val,
                "max": max_val,
            }
        )

    result = pd.DataFrame(rows)
    return normalize_result_frame(result, NUMERIC_SUMMARY_CONTRACT)

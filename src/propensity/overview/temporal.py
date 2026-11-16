"""Semantic time coverage profiling for datetime, ordinal, clock, and intervals."""

from __future__ import annotations

import pandas as pd

from propensity.overview.missing import missing_mask
from propensity.overview.models import (
    TIME_SUMMARY_CONTRACT,
    normalize_result_frame,
)
from propensity.schema.spec import DatasetSpec


def _is_invalid_hhmm(val: object) -> bool:
    """Return True if val cannot be interpreted as a valid HHMM clock code."""
    try:
        num = int(val)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return True
    if num < 0 or num > 2359:
        return True
    minute = num % 100
    hour = num // 100
    return minute >= 60 or hour >= 24


def profile_time_coverage(
    frame: pd.DataFrame,
    table_name: str,
    spec: DatasetSpec,
) -> pd.DataFrame:
    """Profile time coverage across datetime, ordinal, clock, and interval encodings."""
    rows: list[dict[str, object]] = []

    # 1. Datetime columns
    for col in spec.datetime_cols:
        if col not in frame.columns:
            continue
        m_mask = missing_mask(frame[col])
        n_missing = int(m_mask.sum())
        non_m = frame.loc[~m_mask, col]
        n_unique_non_missing = int(non_m.nunique(dropna=False))
        c_min = non_m.min() if len(non_m) > 0 else pd.NA
        c_max = non_m.max() if len(non_m) > 0 else pd.NA
        span = (
            (c_max - c_min)
            if (len(non_m) > 0 and pd.notna(c_min) and pd.notna(c_max))
            else pd.NA
        )

        rows.append(
            {
                "table": table_name,
                "column": str(col),
                "kind": "datetime",
                "unit": pd.NA,
                "start_column": pd.NA,
                "end_column": pd.NA,
                "min": c_min,
                "max": c_max,
                "n_unique_non_missing": n_unique_non_missing,
                "span": span,
                "n_missing": n_missing,
                "missing_endpoint_rows": pd.NA,
                "start_after_end_rows": pd.NA,
                "invalid_clock_rows": pd.NA,
            }
        )

    # 2. Ordinal time columns
    for col in spec.ordinal_time_cols:
        if col not in frame.columns:
            continue
        m_mask = missing_mask(frame[col])
        n_missing = int(m_mask.sum())
        non_m = frame.loc[~m_mask, col]
        n_unique_non_missing = int(non_m.nunique(dropna=False))
        c_min = non_m.min() if len(non_m) > 0 else pd.NA
        c_max = non_m.max() if len(non_m) > 0 else pd.NA
        span = (
            (c_max - c_min)
            if (len(non_m) > 0 and pd.notna(c_min) and pd.notna(c_max))
            else pd.NA
        )

        if col == spec.time_col and spec.time_grain:
            unit = spec.time_grain
        elif "day" in col.lower():
            unit = "day"
        elif "week" in col.lower():
            unit = "week"
        else:
            unit = "ordinal"

        rows.append(
            {
                "table": table_name,
                "column": str(col),
                "kind": "ordinal",
                "unit": unit,
                "start_column": pd.NA,
                "end_column": pd.NA,
                "min": c_min,
                "max": c_max,
                "n_unique_non_missing": n_unique_non_missing,
                "span": span,
                "n_missing": n_missing,
                "missing_endpoint_rows": pd.NA,
                "start_after_end_rows": pd.NA,
                "invalid_clock_rows": pd.NA,
            }
        )

    # 3. Clock time columns
    for col in spec.clock_time_cols:
        if col not in frame.columns:
            continue
        m_mask = missing_mask(frame[col])
        n_missing = int(m_mask.sum())
        non_m = frame.loc[~m_mask, col]
        n_unique_non_missing = int(non_m.nunique(dropna=False))
        c_min = non_m.min() if len(non_m) > 0 else pd.NA
        c_max = non_m.max() if len(non_m) > 0 else pd.NA
        invalid_count = sum(_is_invalid_hhmm(v) for v in non_m)

        rows.append(
            {
                "table": table_name,
                "column": str(col),
                "kind": "clock",
                "unit": "HHMM",
                "start_column": pd.NA,
                "end_column": pd.NA,
                "min": c_min,
                "max": c_max,
                "n_unique_non_missing": n_unique_non_missing,
                "span": pd.NA,  # Clock codes are not continuous durations
                "n_missing": n_missing,
                "missing_endpoint_rows": pd.NA,
                "start_after_end_rows": pd.NA,
                "invalid_clock_rows": invalid_count,
            }
        )

    # 4. Interval columns
    for start_c, end_c in spec.interval_cols:
        if start_c not in frame.columns or end_c not in frame.columns:
            continue
        m_start = missing_mask(frame[start_c])
        m_end = missing_mask(frame[end_c])
        missing_endpoint_rows = int((m_start | m_end).sum())

        valid_mask = (~m_start) & (~m_end)
        start_after_end = int(
            (frame.loc[valid_mask, start_c] > frame.loc[valid_mask, end_c]).sum()
        )
        c_min = frame.loc[~m_start, start_c].min() if (~m_start).any() else pd.NA
        c_max = frame.loc[~m_end, end_c].max() if (~m_end).any() else pd.NA
        span = (c_max - c_min) if (pd.notna(c_min) and pd.notna(c_max)) else pd.NA

        unit = (
            "day" if ("day" in start_c.lower() or "day" in end_c.lower()) else "ordinal"
        )

        rows.append(
            {
                "table": table_name,
                "column": pd.NA,
                "kind": "interval",
                "unit": unit,
                "start_column": str(start_c),
                "end_column": str(end_c),
                "min": c_min,
                "max": c_max,
                "n_unique_non_missing": pd.NA,
                "span": span,
                "n_missing": missing_endpoint_rows,
                "missing_endpoint_rows": missing_endpoint_rows,
                "start_after_end_rows": start_after_end,
                "invalid_clock_rows": pd.NA,
            }
        )

    if not rows:
        return normalize_result_frame(
            pd.DataFrame(columns=list(TIME_SUMMARY_CONTRACT.columns)),
            TIME_SUMMARY_CONTRACT,
        )

    result = pd.DataFrame(rows)
    return normalize_result_frame(result, TIME_SUMMARY_CONTRACT)

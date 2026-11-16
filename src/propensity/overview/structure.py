"""Structure and column profiling functions returning contract DataFrames."""

from __future__ import annotations

import pandas as pd

from propensity.overview.models import (
    COLUMN_SUMMARY_CONTRACT,
    TABLE_SUMMARY_CONTRACT,
    normalize_result_frame,
)
from propensity.schema.infer import resolve_schema
from propensity.schema.spec import ColumnRole, DatasetSpec


def profile_table_structure(frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Profile table dimensions and memory usage."""
    n_rows = len(frame)
    n_cols = len(frame.columns)
    memory_bytes = int(frame.memory_usage(deep=True).sum()) if n_cols > 0 else 0

    result = pd.DataFrame(
        [
            {
                "table": table_name,
                "n_rows": n_rows,
                "n_columns": n_cols,
                "memory_bytes": memory_bytes,
            }
        ]
    )
    return normalize_result_frame(result, TABLE_SUMMARY_CONTRACT)


def profile_columns(
    frame: pd.DataFrame,
    table_name: str,
    spec: DatasetSpec | None = None,
) -> pd.DataFrame:
    """Profile columns in source order with inferred semantic roles and flags."""
    if spec is not None:
        spec.validate(frame)

    if len(frame.columns) == 0:
        return normalize_result_frame(
            pd.DataFrame(columns=list(COLUMN_SUMMARY_CONTRACT.columns)),
            COLUMN_SUMMARY_CONTRACT,
        )

    roles = (
        resolve_schema(frame, spec)
        if spec is not None
        else {col: ColumnRole.OTHER for col in frame.columns}
    )

    rows: list[dict[str, object]] = []
    required_cols = set(spec.required_columns) if spec is not None else set()
    identifier_cols = set(spec.identifier_cols) if spec is not None else set()
    grain_cols = set(spec.grain_cols) if spec is not None else set()
    time_cols = set()
    if spec is not None:
        if spec.time_col:
            time_cols.add(spec.time_col)
        time_cols.update(spec.datetime_cols)
        time_cols.update(spec.ordinal_time_cols)
        time_cols.update(spec.clock_time_cols)
        for start_c, end_c in spec.interval_cols:
            time_cols.add(start_c)
            time_cols.add(end_c)

    for col in frame.columns:
        role = roles.get(col, ColumnRole.OTHER)
        is_time = col in time_cols or role in (ColumnRole.TIME, ColumnRole.DATETIME)
        rows.append(
            {
                "table": table_name,
                "column": str(col),
                "dtype": str(frame.dtypes[col]),
                "role": str(role.value),
                "is_required": col in required_cols,
                "is_identifier": col in identifier_cols,
                "is_grain": col in grain_cols,
                "is_time": is_time,
            }
        )

    result = pd.DataFrame(rows)
    return normalize_result_frame(result, COLUMN_SUMMARY_CONTRACT)

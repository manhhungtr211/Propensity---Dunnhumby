"""Categorical distribution profiling similar to df.describe(include=['O', 'category']) for categorical columns."""

from __future__ import annotations

import pandas as pd
from pandas.api import types as ptypes

from propensity.overview.missing import missing_mask
from propensity.overview.models import (
    CATEGORICAL_SUMMARY_CONTRACT,
    normalize_result_frame,
)
from propensity.schema.infer import resolve_schema
from propensity.schema.spec import ColumnRole, DatasetSpec


def profile_categorical(
    frame: pd.DataFrame,
    table_name: str,
    spec: DatasetSpec | None = None,
) -> pd.DataFrame:
    """Profile distribution of categorical columns: count, n_unique, top, freq, top_pct.

    Filters for columns whose semantic role is ColumnRole.CATEGORICAL (if spec is provided),
    or string/object/category columns (excluding identifier, time, and numeric columns).
    """
    if len(frame.columns) == 0:
        return normalize_result_frame(
            pd.DataFrame(columns=list(CATEGORICAL_SUMMARY_CONTRACT.columns)),
            CATEGORICAL_SUMMARY_CONTRACT,
        )

    roles = resolve_schema(frame, spec) if spec is not None else {}

    cat_cols: list[str] = []
    for col in frame.columns:
        dtype = frame.dtypes[col]

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

        role = roles.get(col)
        if role is not None:
            if role == ColumnRole.CATEGORICAL:
                cat_cols.append(col)
        else:
            if ptypes.is_numeric_dtype(dtype) or ptypes.is_datetime64_any_dtype(dtype):
                continue
            if ptypes.is_string_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype) or dtype == "object":
                cat_cols.append(col)

    if not cat_cols:
        return normalize_result_frame(
            pd.DataFrame(columns=list(CATEGORICAL_SUMMARY_CONTRACT.columns)),
            CATEGORICAL_SUMMARY_CONTRACT,
        )

    rows: list[dict[str, object]] = []
    for col in cat_cols:
        series = frame[col]
        m = missing_mask(series)
        valid = series.loc[~m]
        count = len(valid)

        if count == 0:
            n_unique: object = 0
            top: object = pd.NA
            top_freq: object = pd.NA
            top_pct: object = pd.NA
        else:
            vc = valid.value_counts(dropna=False)
            n_unique = len(vc)
            top_val = vc.index[0]
            top_freq = int(vc.iloc[0])
            top_pct = round(float(top_freq / count * 100), 2)
            top = str(top_val)

        rows.append(
            {
                "table": table_name,
                "column": str(col),
                "count": count,
                "n_unique": n_unique,
                "top": top,
                "freq": top_freq,
                "top_pct": top_pct,
            }
        )

    result = pd.DataFrame(rows)
    return normalize_result_frame(result, CATEGORICAL_SUMMARY_CONTRACT)

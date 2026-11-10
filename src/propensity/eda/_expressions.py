"""Internal Polars expressions shared by EDA computations."""

from __future__ import annotations

import polars as pl

from propensity.eda._errors import AnalysisColumnError


def date_expr(column: str, dtype: pl.DataType) -> pl.Expr:
    """Normalize a Date, Datetime, or parseable String column to Date."""
    if dtype == pl.Date:
        return pl.col(column)
    if dtype.base_type() == pl.Datetime:
        return pl.col(column).dt.date()
    if dtype == pl.String:
        return pl.col(column).str.to_datetime(strict=False).dt.date()
    raise AnalysisColumnError(
        column,
        f"có dtype {dtype}; cần Date, Datetime hoặc String",
    )


def missing_condition(
    column: str,
    dtype: pl.DataType,
    tokens: tuple[str, ...],
) -> pl.Expr:
    """Return a Boolean expression for configured missing representations."""
    value = pl.col(column)
    condition = value.is_null()
    if dtype.is_float():
        condition = condition | value.is_nan().fill_null(False)
    if dtype == pl.String:
        normalized = value.str.strip_chars().str.to_lowercase()
        condition = condition | normalized.is_in(list(tokens)).fill_null(False)
    return condition

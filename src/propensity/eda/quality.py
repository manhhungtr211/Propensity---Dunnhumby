"""Dataset quality checks for missing, duplicate, and constant values."""

from __future__ import annotations

from typing import TypedDict

import polars as pl

from propensity.config import DEFAULT_NA_TOKENS
from propensity.eda._expressions import missing_condition
from propensity.schema.spec import DatasetSpec


class QualityReport(TypedDict):
    """Result returned by :func:`describe_quality`."""

    n_rows: int
    n_cols: int
    n_duplicate_rows: int
    duplicate_row_pct: float
    n_missing_cells: int
    missing_cell_pct: float
    n_columns_with_missing: int
    constant_columns: list[str]
    all_missing_columns: list[str]
    column_quality: pl.DataFrame


def describe_quality(
    lf: pl.LazyFrame,
    spec: DatasetSpec | None = None,
) -> QualityReport:
    """Profile missing cells, exact duplicate rows, and constant columns."""
    schema = lf.collect_schema()
    columns = schema.names()
    tokens = spec.na_tokens if spec is not None else DEFAULT_NA_TOKENS
    flat_columns = [column for column in columns if not schema[column].is_nested()]

    expressions: list[pl.Expr] = [
        pl.len().alias("__n_rows"),
        pl.struct(pl.all()).n_unique().alias("__n_unique_rows"),
    ]
    for column in columns:
        expressions.append(
            missing_condition(column, schema[column], tokens)
            .sum()
            .alias(f"{column}||missing"),
        )
    for column in flat_columns:
        expressions.append(
            pl.col(column)
            .filter(~missing_condition(column, schema[column], tokens))
            .n_unique()
            .alias(f"{column}||unique_valid"),
        )

    statistics = lf.select(expressions).collect().row(0, named=True)
    n_rows = int(statistics["__n_rows"])
    n_unique_rows = int(statistics["__n_unique_rows"])
    n_duplicate_rows = n_rows - n_unique_rows
    rows: list[dict[str, str | int | float | bool | None]] = []
    for column in columns:
        n_missing = int(statistics[f"{column}||missing"])
        n_unique_valid = (
            int(statistics[f"{column}||unique_valid"])
            if column in flat_columns
            else None
        )
        rows.append(
            {
                "column": column,
                "dtype": str(schema[column]),
                "n_missing": n_missing,
                "missing_pct": (100.0 * n_missing / n_rows if n_rows > 0 else 0.0),
                "n_unique_non_missing": n_unique_valid,
                "is_constant": n_unique_valid == 1,
                "is_all_missing": n_rows > 0 and n_missing == n_rows,
            },
        )

    column_quality = pl.DataFrame(
        rows,
        schema={
            "column": pl.String,
            "dtype": pl.String,
            "n_missing": pl.Int64,
            "missing_pct": pl.Float64,
            "n_unique_non_missing": pl.Int64,
            "is_constant": pl.Boolean,
            "is_all_missing": pl.Boolean,
        },
    )
    n_missing_cells = int(column_quality["n_missing"].sum()) if columns else 0
    n_cells = n_rows * len(columns)

    return {
        "n_rows": n_rows,
        "n_cols": len(columns),
        "n_duplicate_rows": n_duplicate_rows,
        "duplicate_row_pct": (100.0 * n_duplicate_rows / n_rows if n_rows > 0 else 0.0),
        "n_missing_cells": n_missing_cells,
        "missing_cell_pct": (100.0 * n_missing_cells / n_cells if n_cells > 0 else 0.0),
        "n_columns_with_missing": column_quality.filter(pl.col("n_missing") > 0).height,
        "constant_columns": column_quality.filter(pl.col("is_constant"))[
            "column"
        ].to_list(),
        "all_missing_columns": column_quality.filter(pl.col("is_all_missing"))[
            "column"
        ].to_list(),
        "column_quality": column_quality,
    }

"""Cardinality summaries for raw features."""

from __future__ import annotations

import polars as pl

from propensity.eda.profiling.models import CARDINALITY_COLUMNS


def summarize_cardinality(
    lf: pl.LazyFrame,
    columns: tuple[str, ...],
) -> pl.DataFrame:
    """Return direct null and distinct-value counts."""
    output_schema = {
        "column": pl.String,
        "n_null": pl.Int64,
        "n_unique": pl.Int64,
    }
    if not columns:
        return pl.DataFrame(schema=output_schema)

    expressions: list[pl.Expr] = []
    for column in columns:
        expressions.extend(
            (
                pl.col(column).null_count().alias(f"{column}__n_null"),
                pl.col(column).drop_nulls().n_unique().alias(f"{column}__n_unique"),
            )
        )
    statistics = lf.select(expressions).collect().row(0, named=True)
    rows = []
    for column in columns:
        n_null = int(statistics[f"{column}__n_null"])
        n_unique = int(statistics[f"{column}__n_unique"])
        rows.append(
            (
                column,
                n_null,
                n_unique,
            )
        )
    return pl.DataFrame(rows, schema=output_schema, orient="row").select(
        CARDINALITY_COLUMNS
    )

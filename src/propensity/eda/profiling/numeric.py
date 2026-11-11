"""Distribution summaries for declared numeric measures."""

from __future__ import annotations

import math
from typing import Final

import polars as pl

from propensity.eda.profiling.models import NUMERIC_PROFILE_COLUMNS
from propensity.schema import AnalysisKind

NUMERIC_SCHEMA: Final = pl.Schema(
    {
        "column": pl.String,
        "min": pl.Float64,
        "max": pl.Float64,
        "mean": pl.Float64,
        "std": pl.Float64,
        "skewness": pl.Float64,
        "p01": pl.Float64,
        "p25": pl.Float64,
        "p50": pl.Float64,
        "p75": pl.Float64,
        "p99": pl.Float64,
        "iqr": pl.Float64,
        "lower_fence": pl.Float64,
        "upper_fence": pl.Float64,
        "n_below_lower_fence": pl.Int64,
        "n_above_upper_fence": pl.Int64,
    }
)


def _finite(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


def profile_numeric_features(
    lf: pl.LazyFrame,
    catalog: pl.DataFrame,
) -> pl.DataFrame:
    """Profile only features declared as numeric measures."""
    columns = catalog.filter(pl.col("analysis_kind") == AnalysisKind.NUMERIC.value)[
        "column"
    ].to_list()
    if not columns:
        return pl.DataFrame(schema=NUMERIC_SCHEMA)

    expressions = []
    for column in columns:
        values = pl.col(column).cast(pl.Float64)
        expressions.extend(
            (
                values.min().alias(f"{column}__min"),
                values.max().alias(f"{column}__max"),
                values.mean().alias(f"{column}__mean"),
                values.std().alias(f"{column}__std"),
                values.skew().alias(f"{column}__skewness"),
                values.quantile(0.01).alias(f"{column}__p01"),
                values.quantile(0.25).alias(f"{column}__p25"),
                values.quantile(0.50).alias(f"{column}__p50"),
                values.quantile(0.75).alias(f"{column}__p75"),
                values.quantile(0.99).alias(f"{column}__p99"),
            )
        )
    statistics = lf.select(expressions).collect().row(0, named=True)

    fences: dict[str, tuple[float | None, float | None]] = {}
    tail_expressions = []
    for column in columns:
        p25 = _finite(statistics[f"{column}__p25"])
        p75 = _finite(statistics[f"{column}__p75"])
        lower = None if p25 is None or p75 is None else p25 - 1.5 * (p75 - p25)
        upper = None if p25 is None or p75 is None else p75 + 1.5 * (p75 - p25)
        fences[column] = (lower, upper)
        if lower is None or upper is None:
            tail_expressions.extend(
                (
                    pl.lit(0, dtype=pl.Int64).alias(f"{column}__below"),
                    pl.lit(0, dtype=pl.Int64).alias(f"{column}__above"),
                )
            )
        else:
            tail_expressions.extend(
                (
                    (pl.col(column) < lower).sum().alias(f"{column}__below"),
                    (pl.col(column) > upper).sum().alias(f"{column}__above"),
                )
            )
    tails = lf.select(tail_expressions).collect().row(0, named=True)

    rows = []
    for column in columns:
        p25 = _finite(statistics[f"{column}__p25"])
        p75 = _finite(statistics[f"{column}__p75"])
        lower, upper = fences[column]
        rows.append(
            (
                column,
                _finite(statistics[f"{column}__min"]),
                _finite(statistics[f"{column}__max"]),
                _finite(statistics[f"{column}__mean"]),
                _finite(statistics[f"{column}__std"]),
                _finite(statistics[f"{column}__skewness"]),
                _finite(statistics[f"{column}__p01"]),
                p25,
                _finite(statistics[f"{column}__p50"]),
                p75,
                _finite(statistics[f"{column}__p99"]),
                None if p25 is None or p75 is None else p75 - p25,
                lower,
                upper,
                int(tails[f"{column}__below"]),
                int(tails[f"{column}__above"]),
            )
        )
    return pl.DataFrame(rows, schema=NUMERIC_SCHEMA, orient="row").select(
        NUMERIC_PROFILE_COLUMNS
    )

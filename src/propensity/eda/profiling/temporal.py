"""Range and coverage summaries for declared temporal features."""

from __future__ import annotations

from typing import Final, assert_never

import polars as pl

from propensity.eda._errors import AnalysisConfigurationError
from propensity.eda.profiling.clock import valid_clock_expr
from propensity.eda.profiling.models import TEMPORAL_PROFILE_COLUMNS
from propensity.schema import AnalysisKind

TEMPORAL_SCHEMA: Final = pl.Schema(
    {
        "column": pl.String,
        "analysis_kind": pl.String,
        "n_valid": pl.Int64,
        "n_unique": pl.Int64,
        "observed_min": pl.String,
        "observed_max": pl.String,
        "span_size": pl.Int64,
        "observed_periods": pl.Int64,
        "coverage_ratio": pl.Float64,
    }
)


def _values_for_kind(column: str, kind: AnalysisKind) -> pl.Expr:
    values = pl.col(column).drop_nulls()
    match kind:
        case AnalysisKind.ORDINAL_TIME:
            return values
        case AnalysisKind.CLOCK_TIME:
            return values.filter(valid_clock_expr(column))
        case AnalysisKind.DATETIME:
            return values
        case AnalysisKind.IDENTIFIER | AnalysisKind.NUMERIC | AnalysisKind.CATEGORICAL:
            raise AnalysisConfigurationError(
                f"analysis_kind '{kind.value}' không phải temporal"
            )
        case unreachable:
            assert_never(unreachable)


def _temporal_metrics(
    kind: AnalysisKind,
    observed_min: int | None,
    observed_max: int | None,
    n_unique: int,
) -> tuple[int | None, int | None, float | None]:
    match kind:
        case AnalysisKind.ORDINAL_TIME:
            if observed_min is None or observed_max is None:
                return None, 0, None
            span_size = observed_max - observed_min + 1
            return span_size, n_unique, n_unique / span_size
        case AnalysisKind.CLOCK_TIME:
            return None, None, None
        case AnalysisKind.DATETIME:
            return None, n_unique, None
        case AnalysisKind.IDENTIFIER | AnalysisKind.NUMERIC | AnalysisKind.CATEGORICAL:
            raise AnalysisConfigurationError(
                f"analysis_kind '{kind.value}' không phải temporal"
            )
        case unreachable:
            assert_never(unreachable)


def profile_temporal_features(
    lf: pl.LazyFrame,
    catalog: pl.DataFrame,
) -> pl.DataFrame:
    """Profile ordinal indices and integer-encoded clock times."""
    temporal_catalog = catalog.filter(
        pl.col("analysis_kind").is_in(
            [
                AnalysisKind.DATETIME.value,
                AnalysisKind.ORDINAL_TIME.value,
                AnalysisKind.CLOCK_TIME.value,
            ]
        )
    )
    columns = temporal_catalog["column"].to_list()
    kinds = [
        AnalysisKind(value) for value in temporal_catalog["analysis_kind"].to_list()
    ]
    if not columns:
        return pl.DataFrame(schema=TEMPORAL_SCHEMA)

    expressions = []
    for column, kind in zip(columns, kinds, strict=True):
        values = _values_for_kind(column, kind)
        expressions.extend(
            (
                values.count().alias(f"{column}__n_valid"),
                values.n_unique().alias(f"{column}__n_unique"),
                values.min().alias(f"{column}__min"),
                values.max().alias(f"{column}__max"),
            )
        )
    statistics = lf.select(expressions).collect().row(0, named=True)
    rows = []
    for column, kind in zip(columns, kinds, strict=True):
        observed_min = statistics[f"{column}__min"]
        observed_max = statistics[f"{column}__max"]
        n_unique = int(statistics[f"{column}__n_unique"])
        span_size, observed_periods, coverage_ratio = _temporal_metrics(
            kind, observed_min, observed_max, n_unique
        )
        rows.append(
            (
                column,
                kind.value,
                int(statistics[f"{column}__n_valid"]),
                n_unique,
                None if observed_min is None else str(observed_min),
                None if observed_max is None else str(observed_max),
                span_size,
                observed_periods,
                coverage_ratio,
            )
        )
    return pl.DataFrame(rows, schema=TEMPORAL_SCHEMA, orient="row").select(
        TEMPORAL_PROFILE_COLUMNS
    )

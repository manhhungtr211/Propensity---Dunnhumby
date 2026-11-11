"""Reusable categorical and grouped-count summaries for technical EDA."""

from __future__ import annotations

import polars as pl

from propensity.eda._errors import AnalysisConfigurationError


def categorical_frequencies(
    lf: pl.LazyFrame,
    columns: tuple[str, ...],
    *,
    top_n: int | None = 10,
) -> pl.DataFrame:
    """Count categorical values and return a consistent long-form table."""
    if top_n is not None and top_n <= 0:
        raise AnalysisConfigurationError("top_n phải lớn hơn 0")
    n_rows = int(lf.select(pl.len()).collect().item())
    frames: list[pl.DataFrame] = []
    for column in columns:
        query = (
            lf.group_by(column)
            .agg(pl.len().alias("n_rows"))
            .sort("n_rows", column, descending=[True, False], nulls_last=True)
        )
        if top_n is not None:
            query = query.head(top_n)
        frames.append(
            query.collect()
            .with_row_index("rank", offset=1)
            .select(
                pl.lit(column).alias("column"),
                pl.col(column)
                .cast(pl.String)
                .fill_null("<NULL>")
                .alias("value"),
                pl.col("n_rows"),
                (pl.col("n_rows") / n_rows).alias("share"),
                pl.col("rank"),
            )
        )
    if not frames:
        return pl.DataFrame(
            schema={
                "column": pl.String,
                "value": pl.String,
                "n_rows": pl.UInt32,
                "share": pl.Float64,
                "rank": pl.UInt32,
            }
        )
    return pl.concat(frames, how="vertical")


def describe_group_counts(
    lf: pl.LazyFrame,
    group_columns: tuple[str, ...],
) -> pl.DataFrame:
    """Describe raw row counts per value for each requested grouping column."""
    frames: list[pl.DataFrame] = []
    for column in group_columns:
        grouped = lf.group_by(column).agg(pl.len().alias("count"))
        frames.append(
            grouped.select(
                pl.len().alias("n_groups"),
                pl.col("count").mean().alias("mean"),
                pl.col("count").std().alias("std"),
                pl.col("count").quantile(0.25).alias("p25"),
                pl.col("count").median().alias("p50"),
                pl.col("count").quantile(0.75).alias("p75"),
                pl.col("count").quantile(0.90).alias("p90"),
                pl.col("count").quantile(0.99).alias("p99"),
                pl.col("count").max().alias("max"),
            )
            .with_columns(pl.lit(column).alias("grain"))
            .select("grain", pl.exclude("grain"))
            .collect()
        )
    return pl.concat(frames, how="vertical").with_columns(
        pl.col("n_groups").cast(pl.Int64),
        pl.exclude("grain", "n_groups").cast(pl.Float64),
    )


def describe_group_metric(
    grouped: pl.LazyFrame,
    *,
    grain: str,
    metric: str,
) -> pl.DataFrame:
    """Describe one already-aggregated numeric metric across groups."""
    stats = grouped.select(
        pl.len().alias("n_groups"),
        pl.col(metric).mean().alias("mean"),
        pl.col(metric).std().alias("std"),
        pl.col(metric).quantile(0.25).alias("p25"),
        pl.col(metric).median().alias("p50"),
        pl.col(metric).quantile(0.75).alias("p75"),
        pl.col(metric).quantile(0.90).alias("p90"),
        pl.col(metric).quantile(0.99).alias("p99"),
        pl.col(metric).max().alias("max"),
    ).collect()
    return stats.select(
        pl.lit(grain).alias("grain"),
        pl.lit(metric).alias("metric"),
        pl.col("n_groups").cast(pl.Int64),
        pl.exclude("n_groups").cast(pl.Float64),
    )


def count_overlapping_intervals(
    frame: pl.DataFrame,
    *,
    start_col: str,
    end_col: str,
) -> int:
    """Count unordered pairs of closed intervals that overlap."""
    intervals = frame.select(start_col, end_col).iter_rows()
    values = list(intervals)
    return sum(
        max(left[0], right[0]) <= min(left[1], right[1])
        for index, left in enumerate(values)
        for right in values[index + 1 :]
    )

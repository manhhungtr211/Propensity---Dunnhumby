"""Temporal coverage checks for longitudinal datasets."""

from __future__ import annotations

from datetime import date
from typing import TypedDict

import polars as pl

from propensity.eda._errors import AnalysisColumnError
from propensity.eda._expressions import date_expr
from propensity.schema.spec import DatasetSpec


class TemporalReport(TypedDict):
    """Result returned by :func:`describe_temporal`."""

    time_col: str
    n_records: int
    n_valid_time: int
    n_missing_or_invalid_time: int
    start_date: date | None
    end_date: date | None
    history_months: int
    observed_months: int
    missing_months: list[date]
    monthly_records: pl.DataFrame
    mean_records_per_month: float
    std_records_per_month: float
    monthly_count_cv: float | None
    min_records_per_month: int
    max_records_per_month: int


def describe_temporal(
    lf: pl.LazyFrame,
    spec: DatasetSpec,
) -> TemporalReport:
    """Measure monthly coverage, gaps, and record-count stability."""
    if spec.time_col is None:
        raise AnalysisColumnError(
            "time_col",
            f"chưa được khai báo trong DatasetSpec '{spec.name}'",
        )

    time_col = spec.time_col
    schema = lf.collect_schema()
    if time_col not in schema:
        raise AnalysisColumnError(
            time_col,
            "không tồn tại trong dữ liệu",
        )

    prepared = lf.select(
        date_expr(time_col, schema[time_col]).alias("__date"),
    )
    totals = (
        prepared.select(
            pl.len().alias("n_records"),
            pl.col("__date").count().alias("n_valid_time"),
        )
        .collect()
        .row(0, named=True)
    )
    n_records = int(totals["n_records"])
    n_valid_time = int(totals["n_valid_time"])

    observed = (
        prepared.filter(pl.col("__date").is_not_null())
        .with_columns(pl.col("__date").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(pl.len().alias("n_records"))
        .sort("month")
        .collect()
    )
    if observed.is_empty():
        return {
            "time_col": time_col,
            "n_records": n_records,
            "n_valid_time": 0,
            "n_missing_or_invalid_time": n_records,
            "start_date": None,
            "end_date": None,
            "history_months": 0,
            "observed_months": 0,
            "missing_months": [],
            "monthly_records": observed,
            "mean_records_per_month": 0.0,
            "std_records_per_month": 0.0,
            "monthly_count_cv": None,
            "min_records_per_month": 0,
            "max_records_per_month": 0,
        }

    date_bounds = (
        prepared.select(
            pl.col("__date").min().alias("start_date"),
            pl.col("__date").max().alias("end_date"),
        )
        .collect()
        .row(0, named=True)
    )
    start_date = date_bounds["start_date"]
    end_date = date_bounds["end_date"]
    first_month = observed["month"].item(0)
    last_month = observed["month"].item(-1)
    calendar = pl.DataFrame(
        {
            "month": pl.date_range(
                first_month,
                last_month,
                interval="1mo",
                eager=True,
            ),
        },
    )
    monthly_records = calendar.join(
        observed,
        on="month",
        how="left",
    ).with_columns(pl.col("n_records").fill_null(0))
    monthly_stats = monthly_records.select(
        pl.col("n_records").mean().alias("mean"),
        pl.col("n_records").std(ddof=0).alias("std"),
        pl.col("n_records").min().alias("min"),
        pl.col("n_records").max().alias("max"),
    ).row(0, named=True)
    mean_records = float(monthly_stats["mean"])
    std_records = float(monthly_stats["std"])

    return {
        "time_col": time_col,
        "n_records": n_records,
        "n_valid_time": n_valid_time,
        "n_missing_or_invalid_time": n_records - n_valid_time,
        "start_date": start_date,
        "end_date": end_date,
        "history_months": monthly_records.height,
        "observed_months": observed.height,
        "missing_months": monthly_records.filter(pl.col("n_records") == 0)[
            "month"
        ].to_list(),
        "monthly_records": monthly_records,
        "mean_records_per_month": mean_records,
        "std_records_per_month": std_records,
        "monthly_count_cv": (std_records / mean_records if mean_records > 0 else None),
        "min_records_per_month": int(monthly_stats["min"]),
        "max_records_per_month": int(monthly_stats["max"]),
    }

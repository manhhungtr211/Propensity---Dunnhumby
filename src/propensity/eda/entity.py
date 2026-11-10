"""Entity-history checks for longitudinal datasets."""

from __future__ import annotations

from typing import TypedDict

import polars as pl

from propensity.eda._errors import AnalysisColumnError, AnalysisConfigurationError
from propensity.eda._expressions import date_expr
from propensity.schema.spec import DatasetSpec


class EntityReport(TypedDict):
    """Result returned by :func:`describe_entities`."""

    entity_col: str
    n_unique_entities: int
    n_records_missing_entity: int
    n_repeat_entities: int
    repeat_entity_pct: float
    min_records_per_entity: int
    median_records_per_entity: float
    avg_records_per_entity: float
    max_records_per_entity: int
    n_entities_active_2plus_months: int
    active_2plus_months_pct: float
    min_history_days: int
    min_history_days_gt_0: int | None
    mean_history_days: float
    median_history_days: float
    max_history_days: int
    entity_history: pl.DataFrame


def describe_entities(
    lf: pl.LazyFrame,
    spec: DatasetSpec,
) -> EntityReport:
    """Summarize repeat activity and first-to-last history for each entity."""
    if spec.entity_col is None:
        raise AnalysisConfigurationError(
            f"[{spec.name}] entity analysis yêu cầu entity_col",
        )
    if spec.time_col is None:
        raise AnalysisConfigurationError(
            f"[{spec.name}] entity analysis yêu cầu time_col",
        )

    entity_col = spec.entity_col
    time_col = spec.time_col

    schema = lf.collect_schema()
    if entity_col not in schema:
        raise AnalysisColumnError(
            entity_col,
            "không tồn tại trong dữ liệu",
        )
    if time_col not in schema:
        raise AnalysisColumnError(
            time_col,
            "không tồn tại trong dữ liệu",
        )

    entity_history_with_null = (
        lf.select(
            pl.col(entity_col),
            date_expr(time_col, schema[time_col]).alias("__date"),
        )
        .with_columns(pl.col("__date").dt.truncate("1mo").alias("__month"))
        .group_by(entity_col)
        .agg(
            pl.len().alias("n_records"),
            pl.col("__month").drop_nulls().n_unique().alias("n_active_months"),
            pl.col("__date").min().alias("first_seen"),
            pl.col("__date").max().alias("last_seen"),
        )
        .collect()
    )
    missing_entity = entity_history_with_null.filter(
        pl.col(entity_col).is_null(),
    )
    n_records_missing_entity = (
        int(missing_entity["n_records"].item()) if missing_entity.height == 1 else 0
    )
    entity_history = (
        entity_history_with_null.filter(pl.col(entity_col).is_not_null())
        .with_columns(
            (pl.col("last_seen") - pl.col("first_seen"))
            .dt.total_days()
            .alias("history_days"),
        )
        .sort(entity_col)
    )
    n_entities = entity_history.height
    if n_entities == 0:
        return {
            "entity_col": entity_col,
            "n_unique_entities": 0,
            "n_records_missing_entity": n_records_missing_entity,
            "n_repeat_entities": 0,
            "repeat_entity_pct": 0.0,
            "min_records_per_entity": 0,
            "median_records_per_entity": 0.0,
            "avg_records_per_entity": 0.0,
            "max_records_per_entity": 0,
            "n_entities_active_2plus_months": 0,
            "active_2plus_months_pct": 0.0,
            "min_history_days": 0,
            "min_history_days_gt_0": None,
            "mean_history_days": 0.0,
            "median_history_days": 0.0,
            "max_history_days": 0,
            "entity_history": entity_history,
        }

    statistics = entity_history.select(
        (pl.col("n_records") > 1).sum().alias("n_repeat"),
        pl.col("n_records").min().alias("min_records"),
        pl.col("n_records").median().alias("median_records"),
        pl.col("n_records").mean().alias("avg_records"),
        pl.col("n_records").max().alias("max_records"),
        (pl.col("n_active_months") >= 2).sum().alias("n_active_2plus"),
        pl.col("history_days").min().fill_null(0).alias("min_history_days"),
        pl.col("history_days")
        .filter(pl.col("history_days") > 0)
        .min()
        .alias("min_history_days_gt_0"),
        pl.col("history_days").mean().fill_null(0.0).alias("mean_history_days"),
        pl.col("history_days").median().fill_null(0.0).alias("median_history_days"),
        pl.col("history_days").max().fill_null(0).alias("max_history_days"),
    ).row(0, named=True)
    n_repeat = int(statistics["n_repeat"])
    n_active_2plus = int(statistics["n_active_2plus"])
    min_gt_0 = statistics["min_history_days_gt_0"]

    return {
        "entity_col": entity_col,
        "n_unique_entities": n_entities,
        "n_records_missing_entity": n_records_missing_entity,
        "n_repeat_entities": n_repeat,
        "repeat_entity_pct": 100.0 * n_repeat / n_entities,
        "min_records_per_entity": int(statistics["min_records"]),
        "median_records_per_entity": float(statistics["median_records"]),
        "avg_records_per_entity": float(statistics["avg_records"]),
        "max_records_per_entity": int(statistics["max_records"]),
        "n_entities_active_2plus_months": n_active_2plus,
        "active_2plus_months_pct": 100.0 * n_active_2plus / n_entities,
        "min_history_days": int(statistics["min_history_days"]),
        "min_history_days_gt_0": (int(min_gt_0) if min_gt_0 is not None else None),
        "mean_history_days": float(statistics["mean_history_days"]),
        "median_history_days": float(statistics["median_history_days"]),
        "max_history_days": int(statistics["max_history_days"]),
        "entity_history": entity_history,
    }

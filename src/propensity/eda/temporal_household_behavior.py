"""Household-level temporal features used by notebook 04."""

from __future__ import annotations

import polars as pl

from propensity.eda.transaction_models import TransactionProfileConfig


def build_household_lifecycle(
    transactions: pl.LazyFrame,
    config: TransactionProfileConfig,
    *,
    snapshot_day: int | None = None,
) -> pl.DataFrame:
    """Build entry timing, exposure, and basket volume at household grain."""
    effective_snapshot = (
        int(
            transactions.select(pl.col(config.time_col).max())
            .collect()
            .item()
        )
        if snapshot_day is None
        else snapshot_day
    )
    eligible = transactions.filter(pl.col(config.time_col) <= effective_snapshot)
    return (
        eligible.group_by(config.entity_col)
        .agg(
            pl.col(config.time_col).min().alias("first_purchase_day"),
            pl.col(config.time_col).max().alias("last_purchase_day"),
            pl.col(config.period_col)
            .sort_by(config.time_col)
            .first()
            .alias("first_purchase_week"),
            pl.col(config.period_col)
            .sort_by(config.time_col)
            .last()
            .alias("last_purchase_week"),
            pl.col(config.basket_col).n_unique().alias("basket_count"),
        )
        .with_columns(
            pl.lit(effective_snapshot).alias("snapshot_day"),
            (
                pl.lit(effective_snapshot)
                - pl.col("first_purchase_day")
                + 1
            ).alias("exposure_days"),
            (
                pl.lit(effective_snapshot) - pl.col("first_purchase_day")
            ).alias("days_since_first_purchase"),
            (
                pl.lit(effective_snapshot) - pl.col("last_purchase_day")
            ).alias("recency_days"),
        )
        .sort(config.entity_col)
        .collect()
    )


def build_household_interpurchase_features(
    transactions: pl.LazyFrame,
    config: TransactionProfileConfig,
    *,
    snapshot_day: int | None = None,
) -> pl.DataFrame:
    """Build household cadence, regularity, and snapshot-lateness features."""
    lifecycle = build_household_lifecycle(
        transactions,
        config,
        snapshot_day=snapshot_day,
    )
    effective_snapshot = int(lifecycle["snapshot_day"][0])
    intervals = (
        transactions.filter(pl.col(config.time_col) <= effective_snapshot)
        .select(config.entity_col, config.time_col)
        .unique()
        .sort(config.entity_col, config.time_col)
        .with_columns(
            pl.col(config.time_col)
            .diff()
            .over(config.entity_col)
            .alias("interpurchase_interval_days")
        )
        .filter(pl.col("interpurchase_interval_days").is_not_null())
    )
    household_intervals = (
        intervals.group_by(config.entity_col)
        .agg(
            pl.len().alias("n_intervals"),
            pl.col("interpurchase_interval_days")
            .median()
            .alias("median_interval_days"),
            pl.col("interpurchase_interval_days")
            .mean()
            .alias("mean_interval_days"),
            pl.col("interpurchase_interval_days")
            .std()
            .alias("std_interval_days"),
            pl.col("interpurchase_interval_days")
            .sort_by(config.time_col)
            .last()
            .alias("last_interval_days"),
        )
        .with_columns(
            (
                pl.col("std_interval_days") / pl.col("mean_interval_days")
            ).alias("interval_cv"),
            (
                pl.col("last_interval_days") / pl.col("median_interval_days")
            ).alias("interval_ratio_last_vs_hist"),
        )
        .collect()
    )
    return (
        lifecycle.join(
            household_intervals,
            on=config.entity_col,
            how="left",
        )
        .with_columns(
            (
                pl.col("recency_days") / pl.col("median_interval_days")
            ).alias("recency_ratio")
        )
        .sort(config.entity_col)
    )

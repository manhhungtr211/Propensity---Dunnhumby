"""Long-history versus recent household behavior features."""

from __future__ import annotations

import polars as pl

from propensity.eda.transaction_models import TransactionProfileConfig


def build_household_behavior_shift(
    transactions: pl.LazyFrame,
    config: TransactionProfileConfig,
    *,
    recent_days: int = 30,
    history_days: int = 365,
    cutoff_day: int | None = None,
) -> pl.DataFrame:
    """Compare non-overlapping historical and recent household behavior."""
    effective_cutoff = (
        int(
            transactions.select(pl.col(config.time_col).max())
            .collect()
            .item()
        )
        if cutoff_day is None
        else cutoff_day
    )
    recent_start = effective_cutoff - recent_days + 1
    history_end = recent_start - 1
    history_start = history_end - history_days + 1
    eligible = (
        transactions.filter(pl.col(config.time_col) <= effective_cutoff)
        .select(config.entity_col)
        .unique()
        .collect()
    )
    baskets = (
        transactions.filter(
            pl.col(config.time_col).is_between(history_start, effective_cutoff)
        )
        .group_by(config.basket_col)
        .agg(
            pl.col(config.entity_col).first(),
            pl.col(config.time_col).first(),
            pl.col(config.product_col).n_unique().alias("n_products"),
        )
        .collect()
    )

    def summarize_period(start_day: int, end_day: int, suffix: str) -> pl.DataFrame:
        return (
            baskets.filter(pl.col(config.time_col).is_between(start_day, end_day))
            .group_by(config.entity_col)
            .agg(
                pl.len().alias(f"n_baskets_{suffix}"),
                pl.col("n_products")
                .median()
                .alias(f"median_products_per_basket_{suffix}"),
            )
        )

    historical = summarize_period(history_start, history_end, "historical")
    recent = summarize_period(recent_start, effective_cutoff, "recent")
    result = (
        eligible.join(historical, on=config.entity_col, how="left")
        .join(recent, on=config.entity_col, how="left")
        .with_columns(
            pl.col("n_baskets_historical").fill_null(0),
            pl.col("n_baskets_recent").fill_null(0),
            pl.lit(effective_cutoff).alias("cutoff_day"),
            pl.lit(history_start).alias("history_start_day"),
            pl.lit(history_end).alias("history_end_day"),
            pl.lit(recent_start).alias("recent_start_day"),
        )
        .with_columns(
            (
                pl.col("n_baskets_historical") / history_days * recent_days
            ).alias("historical_basket_rate_30d"),
            pl.col("n_baskets_recent")
            .cast(pl.Float64)
            .alias("recent_basket_rate_30d"),
        )
        .with_columns(
            pl.when(pl.col("historical_basket_rate_30d") > 0)
            .then(
                pl.col("recent_basket_rate_30d")
                / pl.col("historical_basket_rate_30d")
            )
            .otherwise(None)
            .alias("frequency_ratio"),
            (
                pl.col("median_products_per_basket_recent")
                / pl.col("median_products_per_basket_historical")
            ).alias("basket_size_ratio"),
        )
        .with_columns(
            pl.when(pl.col("n_baskets_recent") == 0)
            .then(pl.lit("Dormant"))
            .when(pl.col("n_baskets_historical") == 0)
            .then(pl.lit("Newly active"))
            .when(pl.col("frequency_ratio") < 0.8)
            .then(pl.lit("Cooling"))
            .when(pl.col("frequency_ratio") <= 1.25)
            .then(pl.lit("Stable"))
            .otherwise(pl.lit("Accelerating"))
            .alias("frequency_state")
        )
    )
    return result.sort(config.entity_col)

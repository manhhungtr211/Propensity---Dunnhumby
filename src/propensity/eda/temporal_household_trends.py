"""Rolling-window household behavior used by notebook 04."""

from __future__ import annotations

import polars as pl

from propensity.eda._errors import AnalysisConfigurationError
from propensity.eda.transaction_models import TransactionProfileConfig


def build_household_rolling_windows(
    transactions: pl.LazyFrame,
    config: TransactionProfileConfig,
    *,
    window_weeks: int = 13,
    step_weeks: int = 4,
) -> pl.DataFrame:
    """Aggregate household cadence and basket rate in full rolling windows."""
    if window_weeks <= 0 or step_weeks <= 0:
        raise AnalysisConfigurationError(
            "window_weeks và step_weeks phải lớn hơn 0"
        )
    period_bounds = (
        transactions.select(
            pl.col(config.period_col).min().alias("min_week"),
            pl.col(config.period_col).max().alias("max_week"),
        )
        .collect()
        .row(0, named=True)
    )
    min_week = int(period_bounds["min_week"])
    max_week = int(period_bounds["max_week"])
    final_start = max_week - window_weeks + 1
    if final_start < min_week:
        raise AnalysisConfigurationError(
            "Observation window ngắn hơn window_weeks"
        )
    starts = list(range(min_week, final_start + 1, step_weeks))
    starts.extend((max(min_week, final_start - window_weeks), final_start))
    window_starts = sorted(set(starts))

    baskets = (
        transactions.select(
            config.entity_col,
            config.basket_col,
            config.period_col,
        )
        .unique()
        .collect()
    )
    interval_events = (
        transactions.select(
            config.entity_col,
            config.time_col,
            config.period_col,
        )
        .unique(subset=[config.entity_col, config.time_col])
        .sort(config.entity_col, config.time_col)
        .with_columns(
            pl.col(config.time_col)
            .diff()
            .over(config.entity_col)
            .alias("interval_days")
        )
        .filter(pl.col("interval_days").is_not_null())
        .collect()
    )

    frames: list[pl.DataFrame] = []
    for start_week in window_starts:
        end_week = start_week + window_weeks - 1
        basket_window = (
            baskets.filter(
                pl.col(config.period_col).is_between(start_week, end_week)
            )
            .group_by(config.entity_col)
            .agg(pl.len().alias("n_baskets"))
            .with_columns(
                (pl.col("n_baskets") / window_weeks).alias(
                    "baskets_per_week"
                )
            )
        )
        interval_window = (
            interval_events.filter(
                pl.col(config.period_col).is_between(start_week, end_week)
            )
            .group_by(config.entity_col)
            .agg(
                pl.len().alias("n_intervals"),
                pl.col("interval_days").median().alias("median_interval_days"),
                pl.col("interval_days").mean().alias("mean_interval_days"),
                pl.col("interval_days").std().alias("std_interval_days"),
            )
            .with_columns(
                (
                    pl.col("std_interval_days")
                    / pl.col("mean_interval_days")
                ).alias("interval_cv")
            )
        )
        frames.append(
            basket_window.join(
                interval_window,
                on=config.entity_col,
                how="left",
            ).with_columns(
                pl.lit(start_week).alias("window_start_week"),
                pl.lit(end_week).alias("window_end_week"),
                pl.lit(window_weeks).alias("window_weeks"),
            )
        )
    return pl.concat(frames, how="diagonal_relaxed").sort(
        "window_end_week",
        config.entity_col,
    )

"""Historical activity coverage at a fixed household cutoff."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from propensity.eda._errors import AnalysisConfigurationError
from propensity.eda.transaction_models import TransactionProfileConfig


def profile_historical_activity_coverage(
    transactions: pl.LazyFrame,
    config: TransactionProfileConfig,
    *,
    windows_days: Sequence[int] = (3, 7, 14, 21, 30, 60, 90, 180, 365),
    cutoff_day: int | None = None,
) -> pl.DataFrame:
    """Measure usable household history for several trailing windows."""
    if not windows_days or any(window <= 0 for window in windows_days):
        raise AnalysisConfigurationError(
            "windows_days phải chứa các số nguyên dương"
        )
    effective_cutoff = (
        int(
            transactions.select(pl.col(config.time_col).max())
            .collect()
            .item()
        )
        if cutoff_day is None
        else cutoff_day
    )
    eligible = (
        transactions.filter(pl.col(config.time_col) <= effective_cutoff)
        .select(config.entity_col)
        .unique()
        .collect()
    )
    n_households = eligible.height
    rows: list[dict[str, int | float]] = []
    for window_days in windows_days:
        start_day = effective_cutoff - window_days + 1
        activity = (
            transactions.filter(
                pl.col(config.time_col).is_between(start_day, effective_cutoff)
            )
            .group_by(config.entity_col)
            .agg(
                pl.col(config.basket_col).n_unique().alias("n_baskets"),
                pl.col(config.time_col).n_unique().alias("n_active_days"),
                pl.col(config.period_col).n_unique().alias("n_active_weeks"),
            )
            .collect()
        )
        household_activity = eligible.join(
            activity,
            on=config.entity_col,
            how="left",
        ).with_columns(
            pl.col("n_baskets").fill_null(0),
            pl.col("n_active_days").fill_null(0),
            pl.col("n_active_weeks").fill_null(0),
        )
        rows.append(
            {
                "cutoff_day": effective_cutoff,
                "window_days": window_days,
                "window_start_day": start_day,
                "n_eligible_households": n_households,
                "share_ge_1_basket": float(
                    household_activity.select(
                        (pl.col("n_baskets") >= 1).mean()
                    ).item()
                ),
                "share_ge_2_baskets": float(
                    household_activity.select(
                        (pl.col("n_baskets") >= 2).mean()
                    ).item()
                ),
                "share_ge_5_baskets": float(
                    household_activity.select(
                        (pl.col("n_baskets") >= 5).mean()
                    ).item()
                ),
                "median_basket_count": float(
                    household_activity["n_baskets"].median()
                ),
                "share_gap_ready": float(
                    household_activity.select(
                        (pl.col("n_active_days") >= 2).mean()
                    ).item()
                ),
                "share_ge_2_active_weeks": float(
                    household_activity.select(
                        (pl.col("n_active_weeks") >= 2).mean()
                    ).item()
                ),
            }
        )
    return pl.DataFrame(rows).sort("window_days")

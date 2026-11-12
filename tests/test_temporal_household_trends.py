from __future__ import annotations

from typing import Final

import matplotlib
import polars as pl
from matplotlib.figure import Figure

from propensity.eda.temporal_household_trends import (
    build_household_rolling_windows,
)
from propensity.eda.transaction_models import TransactionProfileConfig
from propensity.eda.visualization import plot_household_behavior_change

matplotlib.use("Agg")

TEST_CONFIG: Final = TransactionProfileConfig(
    entity_col="household_key",
    basket_col="BASKET_ID",
    time_col="DAY",
    period_col="WEEK_NO",
    product_col="PRODUCT_ID",
    store_col="STORE_ID",
    quantity_col="QUANTITY",
    sales_col="SALES_VALUE",
    discount_cols=("RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC"),
)


def _transactions() -> pl.LazyFrame:
    households: list[int] = []
    baskets: list[int] = []
    days: list[int] = []
    weeks: list[int] = []
    for household in range(1, 7):
        for week in range(1, 27):
            households.append(household)
            baskets.append(household * 1000 + week)
            days.append((week - 1) * 7 + household)
            weeks.append(week)
    row_count = len(households)
    return pl.LazyFrame(
        {
            "household_key": households,
            "BASKET_ID": baskets,
            "DAY": days,
            "WEEK_NO": weeks,
            "PRODUCT_ID": [10] * row_count,
            "STORE_ID": [7] * row_count,
            "QUANTITY": [1] * row_count,
            "SALES_VALUE": [1.0] * row_count,
            "RETAIL_DISC": [0.0] * row_count,
            "COUPON_DISC": [0.0] * row_count,
            "COUPON_MATCH_DISC": [0.0] * row_count,
        }
    )


def test_rolling_windows_keep_full_latest_and_prior_periods() -> None:
    # Given / When
    result = build_household_rolling_windows(
        _transactions(),
        TEST_CONFIG,
        window_weeks=13,
        step_weeks=4,
    )

    # Then
    starts = result["window_start_week"].unique().sort().to_list()
    assert 1 in starts
    assert 14 in starts
    assert starts[-1] == 14
    assert result["window_end_week"].max() == 26
    assert result.filter(pl.col("window_start_week") == 14).height == 6


def test_behavior_change_figure_has_trend_comparison_and_states() -> None:
    # Given
    windows = build_household_rolling_windows(
        _transactions(),
        TEST_CONFIG,
        window_weeks=13,
        step_weeks=4,
    )

    # When
    figure: Figure = plot_household_behavior_change(windows)

    # Then
    assert len(figure.axes) == 3
    assert figure.axes[1].get_xscale() == "log"
    assert figure.axes[1].get_yscale() == "log"

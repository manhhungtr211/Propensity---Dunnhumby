from __future__ import annotations

from typing import Final

import matplotlib
import polars as pl
from matplotlib.figure import Figure

from propensity.eda.temporal_activity_coverage import (
    profile_historical_activity_coverage,
)
from propensity.eda.transaction_models import TransactionProfileConfig
from propensity.eda.visualization import plot_historical_activity_coverage

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
    return pl.LazyFrame(
        {
            "household_key": [1, 1, 1, 2, 2, 3],
            "BASKET_ID": [101, 102, 103, 201, 202, 301],
            "DAY": [80, 90, 100, 20, 95, 40],
            "WEEK_NO": [12, 13, 15, 3, 14, 6],
            "PRODUCT_ID": [10] * 6,
            "STORE_ID": [7] * 6,
            "QUANTITY": [1] * 6,
            "SALES_VALUE": [1.0] * 6,
            "RETAIL_DISC": [0.0] * 6,
            "COUPON_DISC": [0.0] * 6,
            "COUPON_MATCH_DISC": [0.0] * 6,
        }
    )


def test_coverage_keeps_inactive_households_as_zero() -> None:
    # Given / When
    result = profile_historical_activity_coverage(
        _transactions(),
        TEST_CONFIG,
        windows_days=(30, 90),
        cutoff_day=100,
    )

    # Then
    short = result.filter(pl.col("window_days") == 30).row(0, named=True)
    assert short["n_eligible_households"] == 3
    assert short["share_ge_1_basket"] == 2 / 3
    assert short["share_ge_2_baskets"] == 1 / 3
    assert short["share_gap_ready"] == 1 / 3
    assert short["median_basket_count"] == 1


def test_coverage_figure_has_heatmap_colorbar_and_median_panel() -> None:
    # Given
    coverage = profile_historical_activity_coverage(
        _transactions(),
        TEST_CONFIG,
        windows_days=(30, 90),
        cutoff_day=100,
    )

    # When
    figure: Figure = plot_historical_activity_coverage(coverage)

    # Then
    assert len(figure.axes) == 3

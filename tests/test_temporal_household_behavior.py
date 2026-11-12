from __future__ import annotations

from typing import Final

import matplotlib
import polars as pl
from matplotlib.figure import Figure

from propensity.eda.temporal_household_behavior import (
    build_household_interpurchase_features,
    build_household_lifecycle,
)
from propensity.eda.transaction_models import TransactionProfileConfig
from propensity.eda.visualization import (
    plot_household_interpurchase_dynamics,
    plot_last_purchase_recency,
    plot_lifecycle_coverage_exposure,
)

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
            "BASKET_ID": [100, 100, 101, 200, 201, 300],
            "DAY": [1, 1, 8, 5, 20, 15],
            "WEEK_NO": [1, 1, 2, 1, 3, 3],
            "PRODUCT_ID": [10, 11, 12, 10, 13, 14],
            "STORE_ID": [7, 7, 8, 7, 9, 8],
            "QUANTITY": [1, 1, 1, 1, 1, 1],
            "SALES_VALUE": [5.0, 3.0, 4.0, 8.0, 12.0, 2.0],
            "RETAIL_DISC": [0.0] * 6,
            "COUPON_DISC": [0.0] * 6,
            "COUPON_MATCH_DISC": [0.0] * 6,
        }
    )


def test_lifecycle_uses_distinct_baskets_and_snapshot_exposure() -> None:
    # Given / When
    result = build_household_lifecycle(
        _transactions(),
        TEST_CONFIG,
        snapshot_day=15,
    )

    # Then
    first = result.filter(pl.col("household_key") == 1).row(0, named=True)
    assert first["first_purchase_day"] == 1
    assert first["first_purchase_week"] == 1
    assert first["last_purchase_week"] == 2
    assert first["basket_count"] == 2
    assert first["exposure_days"] == 15
    assert first["recency_days"] == 7
    assert result["household_key"].to_list() == [1, 2, 3]


def test_lifecycle_ignores_transactions_after_snapshot() -> None:
    # Given / When
    result = build_household_lifecycle(
        _transactions(),
        TEST_CONFIG,
        snapshot_day=15,
    )

    # Then
    second = result.filter(pl.col("household_key") == 2).row(0, named=True)
    assert second["basket_count"] == 1
    assert second["last_purchase_day"] == 5


def test_lifecycle_figure_has_two_question_driven_panels() -> None:
    # Given
    households = build_household_lifecycle(_transactions(), TEST_CONFIG)

    # When
    figure: Figure = plot_lifecycle_coverage_exposure(households)

    # Then
    assert len(figure.axes) == 2
    assert figure.axes[1].get_yscale() == "log"


def test_last_purchase_figure_has_cohort_and_recency_panels() -> None:
    # Given
    households = build_household_lifecycle(_transactions(), TEST_CONFIG)

    # When
    figure: Figure = plot_last_purchase_recency(households)

    # Then
    assert len(figure.axes) == 2
    assert figure.axes[1].get_ylim() == (0.0, 1.02)


def test_interpurchase_features_preserve_household_weighting() -> None:
    # Given / When
    result = build_household_interpurchase_features(_transactions(), TEST_CONFIG)

    # Then
    first = result.filter(pl.col("household_key") == 1).row(0, named=True)
    assert first["n_intervals"] == 1
    assert first["median_interval_days"] == 7
    assert first["last_interval_days"] == 7
    assert first["recency_ratio"] == 12 / 7


def test_interpurchase_figure_has_one_main_and_two_supporting_panels() -> None:
    # Given
    transactions = pl.LazyFrame(
        {
            "household_key": [1, 1, 1, 1, 2, 2, 2, 2],
            "BASKET_ID": [101, 102, 103, 104, 201, 202, 203, 204],
            "DAY": [1, 3, 6, 10, 1, 4, 8, 13],
            "WEEK_NO": [1, 1, 1, 2, 1, 1, 2, 2],
            "PRODUCT_ID": [10] * 8,
            "STORE_ID": [7] * 8,
            "QUANTITY": [1] * 8,
            "SALES_VALUE": [1.0] * 8,
            "RETAIL_DISC": [0.0] * 8,
            "COUPON_DISC": [0.0] * 8,
            "COUPON_MATCH_DISC": [0.0] * 8,
        }
    )
    households = build_household_interpurchase_features(transactions, TEST_CONFIG)
    event_distribution = pl.DataFrame(
        {
            "return_interval_days": [2, 3, 4, 5],
            "n_returns": [1, 2, 2, 1],
            "share": [1 / 6, 2 / 6, 2 / 6, 1 / 6],
            "cumulative_share": [1 / 6, 3 / 6, 5 / 6, 1.0],
        }
    )

    # When
    figure: Figure = plot_household_interpurchase_dynamics(
        households,
        event_distribution,
    )

    # Then
    assert len(figure.axes) == 3

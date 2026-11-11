from __future__ import annotations

from typing import Final

import polars as pl
import pytest

from propensity.eda.transaction import (
    build_transaction_technical_report,
    profile_identifier_frequencies,
    profile_return_intervals,
    summarize_period_activity,
)
from propensity.eda.transaction_models import TransactionProfileConfig

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


@pytest.fixture
def transaction_frame() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "household_key": [1, 1, 1, 1, 2, 2, 2],
            "BASKET_ID": [100, 100, 101, 102, 200, 201, 202],
            "DAY": [1, 1, 3, 8, 2, 2, 5],
            "PRODUCT_ID": [10, 11, 10, 12, 10, 11, 13],
            "QUANTITY": [1, 2, 1, 3, 1, 1, 2],
            "SALES_VALUE": [5.0, 8.0, 4.0, 12.0, 3.0, 2.0, 9.0],
            "STORE_ID": [7, 7, 7, 8, 7, 7, 9],
            "RETAIL_DISC": [-1.0, 0.0, 0.0, -2.0, 0.0, 0.0, -1.0],
            "WEEK_NO": [1, 1, 1, 2, 1, 1, 1],
            "COUPON_DISC": [0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0],
            "COUPON_MATCH_DISC": [0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0],
        }
    )


def test_return_intervals_use_distinct_active_days(
    transaction_frame: pl.LazyFrame,
) -> None:
    # Given / When
    result = profile_return_intervals(transaction_frame, TEST_CONFIG)

    # Then
    assert result["return_interval_days"].to_list() == [2, 3, 5]
    assert result["n_returns"].to_list() == [1, 1, 1]
    assert result["share"].to_list() == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert result["cumulative_share"].to_list() == pytest.approx([1 / 3, 2 / 3, 1])


def test_period_activity_preserves_requested_period(
    transaction_frame: pl.LazyFrame,
) -> None:
    # Given / When
    result = summarize_period_activity(transaction_frame, "DAY", TEST_CONFIG)

    # Then
    day_one = result.filter(pl.col("period_value") == 1).row(0, named=True)
    assert day_one["period"] == "DAY"
    assert day_one["n_product_lines"] == 2
    assert day_one["n_baskets"] == 1
    assert day_one["n_households"] == 1
    assert day_one["total_sales"] == pytest.approx(13.0)
    assert day_one["total_quantity"] == pytest.approx(3.0)


def test_identifier_frequency_counts_raw_product_lines(
    transaction_frame: pl.LazyFrame,
) -> None:
    # Given / When
    result = profile_identifier_frequencies(
        transaction_frame,
        ("STORE_ID",),
        top_n=2,
    )

    # Then
    assert result["value"].to_list() == ["7", "8"]
    assert result["n_rows"].to_list() == [5, 1]
    assert result["share"].to_list() == pytest.approx([5 / 7, 1 / 7])
    assert result["rank"].to_list() == [1, 2]


def test_combined_transaction_report_covers_every_technical_grain(
    transaction_frame: pl.LazyFrame,
) -> None:
    # Given / When
    result = build_transaction_technical_report(
        transaction_frame,
        TEST_CONFIG,
        top_n=2,
    )

    # Then
    assert set(result.identifier_frequencies["column"]) == {
        "household_key",
        "BASKET_ID",
        "PRODUCT_ID",
        "STORE_ID",
    }
    assert result.daily_activity.height == 5
    assert result.period_activity.height == 2
    assert set(result.basket_distributions["metric"]) == {
        "n_product_lines",
        "n_products",
        "basket_sales",
        "basket_quantity",
        "basket_discount",
    }
    assert set(result.household_distributions["metric"]) == {
        "n_product_lines",
        "n_baskets",
        "n_active_days",
        "n_products",
        "n_stores",
        "total_sales",
        "total_quantity",
        "total_discount",
        "span_days",
    }
    assert result.return_intervals["n_returns"].sum() == 3
    assert result.relationships.height == 3

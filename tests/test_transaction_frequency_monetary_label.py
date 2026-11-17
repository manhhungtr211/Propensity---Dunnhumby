from __future__ import annotations

import pandas as pd
import pytest

from propensity.eda.transaction_frequency_monetary_label import (
    build_transaction_frequency_monetary_features,
    summarize_frequency_monetary_by_label,
)


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "household_key": [1, 2],
            "cutoff_day": [10, 10],
            "label": [1, 0],
        }
    )


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "household_key": [1, 1, 1, 1, 1, 2, 2, 2],
            "BASKET_ID": [101, 101, 102, 103, 104, 201, 202, 203],
            "DAY": [5, 5, 5, 8, 10, 4, 7, 9],
            "QUANTITY": [1, 1, 1, 1, 1, 1, 1, -1],
            "SALES_VALUE": [10.0, 5.0, 20.0, 30.0, 1_000.0, 50.0, 10.0, -5.0],
        }
    )


def test_features_use_basket_grain_inside_pre_cutoff_window() -> None:
    # Given / When
    result = build_transaction_frequency_monetary_features(
        _labels(),
        _transactions(),
        observation_days=5,
    )

    # Then
    household_1 = result.loc[result["household_key"] == 1].iloc[0]
    assert household_1["n_baskets_90d"] == 3
    assert household_1["n_active_days_90d"] == 2
    assert household_1["baskets_per_active_day_90d"] == pytest.approx(1.5)
    assert household_1["total_sales_value_90d"] == pytest.approx(65.0)
    assert household_1["avg_spend_per_basket_90d"] == pytest.approx(65 / 3)
    assert household_1["median_basket_value_90d"] == pytest.approx(20.0)
    assert household_1["max_basket_value_90d"] == pytest.approx(30.0)
    assert household_1["spend_per_active_day_90d"] == pytest.approx(32.5)


def test_invalid_and_out_of_window_lines_are_excluded() -> None:
    # Given / When
    result = build_transaction_frequency_monetary_features(
        _labels(),
        _transactions(),
        observation_days=5,
    )

    # Then
    household_2 = result.loc[result["household_key"] == 2].iloc[0]
    assert household_2["n_baskets_90d"] == 1
    assert household_2["total_sales_value_90d"] == pytest.approx(10.0)
    assert household_2["max_basket_value_90d"] == pytest.approx(10.0)


def test_summary_keeps_negative_and_positive_rows_separate() -> None:
    # Given
    feature_panel = build_transaction_frequency_monetary_features(
        _labels(),
        _transactions(),
        observation_days=5,
    )

    # When
    summary = summarize_frequency_monetary_by_label(feature_panel)

    # Then
    total_sales = summary.loc[summary["feature"] == "total_sales_value_90d"]
    assert total_sales.loc[total_sales["label"] == 1, "median"].iat[0] == 65
    assert total_sales.loc[total_sales["label"] == 0, "median"].iat[0] == 10

from __future__ import annotations

import pandas as pd
import pytest

from propensity.eda.transaction_quantity_basket_label import (
    build_transaction_quantity_basket_features,
    summarize_quantity_basket_by_label,
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
            "household_key": [1, 1, 1, 1, 2, 2, 2],
            "BASKET_ID": [101, 101, 102, 103, 201, 202, 203],
            "DAY": [5, 5, 8, 10, 4, 7, 9],
            "PRODUCT_ID": [11, 12, 11, 13, 21, 22, 23],
            "QUANTITY": [2, 3, 1, 100, 5, 4, -1],
            "SALES_VALUE": [10.0, 15.0, 5.0, 100.0, 25.0, 20.0, -5.0],
        }
    )


def test_features_use_basket_grain_and_pre_cutoff_window() -> None:
    # Given / When
    result = build_transaction_quantity_basket_features(
        _labels(),
        _transactions(),
        observation_days=5,
    )

    # Then
    household_1 = result.loc[result["household_key"] == 1].iloc[0]
    assert household_1["total_quantity_90d"] == 6
    assert household_1["avg_items_per_basket_90d"] == pytest.approx(3.0)
    assert household_1["avg_unique_products_per_basket_90d"] == pytest.approx(1.5)
    assert household_1["unique_products_90d"] == 2


def test_invalid_and_out_of_window_lines_are_excluded() -> None:
    # Given / When
    result = build_transaction_quantity_basket_features(
        _labels(),
        _transactions(),
        observation_days=5,
    )

    # Then
    household_2 = result.loc[result["household_key"] == 2].iloc[0]
    assert household_2["total_quantity_90d"] == 4
    assert household_2["avg_items_per_basket_90d"] == pytest.approx(4.0)
    assert household_2["unique_products_90d"] == 1


def test_summary_separates_negative_and_positive_labels() -> None:
    # Given
    feature_panel = build_transaction_quantity_basket_features(
        _labels(),
        _transactions(),
        observation_days=5,
    )

    # When
    summary = summarize_quantity_basket_by_label(feature_panel)

    # Then
    total_quantity = summary.loc[summary["feature"] == "total_quantity_90d"]
    assert total_quantity.loc[total_quantity["label"] == 1, "median"].iat[0] == 6
    assert total_quantity.loc[total_quantity["label"] == 0, "median"].iat[0] == 4

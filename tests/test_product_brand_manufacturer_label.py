from __future__ import annotations

import pandas as pd

from propensity.eda.product_brand_manufacturer_label import (
    build_product_brand_manufacturer_report,
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
            "household_key": [1, 1, 1, 1, 2, 2],
            "DAY": [5, 6, 8, 10, 7, 9],
            "PRODUCT_ID": [11, 11, 12, 13, 13, 14],
            "QUANTITY": [1, 1, 1, 1, 1, -1],
            "SALES_VALUE": [1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
        }
    )


def _products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PRODUCT_ID": [11, 12, 13, 14],
            "BRAND": ["National", "Private", "Private", "National"],
            "MANUFACTURER": [101, 102, 103, 104],
        }
    )


def test_brand_features_use_only_valid_pre_cutoff_lines() -> None:
    # Given / When
    report = build_product_brand_manufacturer_report(
        _labels(),
        _transactions(),
        _products(),
        observation_days=5,
    )

    # Then
    household_1 = report.feature_panel.loc[
        report.feature_panel["household_key"] == 1
    ].iloc[0]
    assert household_1["n_brands_90d"] == 2
    assert household_1["dominant_brand_90d"] == "National"
    assert household_1["dominant_brand_share_90d"] == 2 / 3


def test_brand_history_is_compared_within_each_label() -> None:
    # Given / When
    report = build_product_brand_manufacturer_report(
        _labels(),
        _transactions(),
        _products(),
        observation_days=5,
    )

    # Then
    private_positive = report.historical_brand_summary.loc[
        (report.historical_brand_summary["category"] == "Private")
        & (report.historical_brand_summary["label"] == 1),
        "within_label_share",
    ].iat[0]
    assert private_positive == 1


def test_invalid_manufacturer_purchase_is_excluded() -> None:
    # Given / When
    report = build_product_brand_manufacturer_report(
        _labels(),
        _transactions(),
        _products(),
        observation_days=5,
    )

    # Then
    assert "104" not in set(report.manufacturer_summary["category"])

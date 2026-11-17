from __future__ import annotations

import pandas as pd

from propensity.eda.product_category_label import (
    build_product_category_label_report,
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
            "DEPARTMENT": ["A", "A", "B", "B"],
            "COMMODITY_DESC": ["X", "Y", "Z", "Z"],
            "SUB_COMMODITY_DESC": ["X1", "Y1", "Z1", "Z2"],
        }
    )


def test_breadth_and_dominant_categories_use_only_pre_cutoff_window() -> None:
    # Given / When
    report = build_product_category_label_report(
        _labels(),
        _transactions(),
        _products(),
        observation_days=5,
    )

    # Then
    household_1 = report.feature_panel.loc[
        report.feature_panel["household_key"] == 1
    ].iloc[0]
    assert household_1["n_departments_90d"] == 1
    assert household_1["n_commodities_90d"] == 2
    assert "n_commoditys_90d" not in report.feature_panel.columns
    assert household_1["dominant_department_90d"] == "A"
    assert household_1["dominant_commodity_90d"] == "X"


def test_historical_category_distribution_is_conditioned_on_label() -> None:
    # Given / When
    report = build_product_category_label_report(
        _labels(),
        _transactions(),
        _products(),
        observation_days=5,
    )

    # Then
    departments = report.historical_category_summary.loc[
        report.historical_category_summary["dimension"] == "department"
    ]
    department_a_positive = departments.loc[
        (departments["category"] == "A") & (departments["label"] == 1),
        "within_label_share",
    ].iat[0]
    department_b_negative = departments.loc[
        (departments["category"] == "B") & (departments["label"] == 0),
        "within_label_share",
    ].iat[0]
    assert department_a_positive == 1
    assert department_b_negative == 1


def test_invalid_product_line_does_not_enter_history() -> None:
    # Given / When
    report = build_product_category_label_report(
        _labels(),
        _transactions(),
        _products(),
        observation_days=5,
    )

    # Then
    subcommodities = set(
        report.historical_category_summary.loc[
            report.historical_category_summary["dimension"] == "subcommodity",
            "category",
        ]
    )
    assert "Z2" not in subcommodities

from __future__ import annotations

import pandas as pd
import pytest

from propensity.eda.transaction_time_label import (
    build_transaction_time_features,
    summarize_transaction_time_by_label,
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
            "DAY": [5, 7, 9, 10, 1, 8],
            "WEEK_NO": [1, 1, 2, 2, 1, 2],
            "QUANTITY": [1, 1, 1, 1, 1, -1],
            "SALES_VALUE": [1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
        }
    )


def test_features_use_only_valid_events_before_cutoff() -> None:
    # Given / When
    result = build_transaction_time_features(
        _labels(),
        _transactions(),
        observation_days=5,
    )

    # Then
    household_1 = result.loc[result["household_key"] == 1].iloc[0]
    assert household_1["days_since_last_purchase"] == 1
    assert household_1["active_days_90d"] == 3
    assert household_1["active_weeks_90d"] == 2
    assert household_1["median_interpurchase_gap_days_90d"] == 2
    assert household_1["interpurchase_gap_std_days_90d"] == 0


def test_household_without_recent_activity_keeps_recency_and_zero_counts() -> None:
    # Given / When
    result = build_transaction_time_features(
        _labels(),
        _transactions(),
        observation_days=5,
    )

    # Then
    household_2 = result.loc[result["household_key"] == 2].iloc[0]
    assert household_2["days_since_last_purchase"] == 9
    assert household_2["active_days_90d"] == 0
    assert household_2["active_weeks_90d"] == 0
    assert pd.isna(household_2["median_interpurchase_gap_days_90d"])


def test_summary_keeps_labels_separate() -> None:
    # Given
    feature_panel = build_transaction_time_features(
        _labels(),
        _transactions(),
        observation_days=5,
    )

    # When
    summary = summarize_transaction_time_by_label(feature_panel)

    # Then
    recency = summary.loc[summary["feature"] == "days_since_last_purchase"]
    assert recency.loc[recency["label"] == 1, "median"].iat[0] == 1
    assert recency.loc[recency["label"] == 0, "median"].iat[0] == 9
    gap = summary.loc[
        (summary["feature"] == "median_interpurchase_gap_days_90d")
        & (summary["label"] == 0)
    ].iloc[0]
    assert gap["missing_rate"] == pytest.approx(1.0)

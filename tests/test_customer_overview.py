"""Unit tests for customer overview metrics, purchases, and visualizations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from propensity.eda.customer_overview import (
    CustomerOverviewError,
    build_household_summary,
    build_purchase_days,
    compute_concentration_stats,
    describe_customer_metrics,
    gini,
    lorenz_curve,
)
from propensity.eda.customer_overview_plots import (
    plot_customer_lorenz_curves,
    plot_customer_overview_histograms,
)
from propensity.purchases import PurchaseValidationError, valid_purchase_lines


def _synthetic_transactions() -> pd.DataFrame:
    """Synthetic transactions with 4 households testing different edge cases."""
    return pd.DataFrame(
        {
            "household_key": [
                1, 1, 1, 1,  # HH 1: 4 trips on 3 days (10, 15, 20), spend=140, span=11
                2, 2,        # HH 2: 2 trips on 2 days (10, 24), spend=70, span=15
                3,           # HH 3: 1 trip on 1 day (50), spend=20, span=1
                4, 4, 4,     # HH 4: 3 trips on 3 days (1, 30, 60), spend=300, span=60
                5,           # HH 5: Non-positive sales (should be filtered out)
            ],
            "BASKET_ID": [
                101, 102, 103, 104,
                201, 202,
                301,
                401, 402, 403,
                501,
            ],
            "DAY": [
                10, 10, 15, 20,
                10, 24,
                50,
                1, 30, 60,
                10,
            ],
            "SALES_VALUE": [
                20.0, 30.0, 40.0, 50.0,
                30.0, 40.0,
                20.0,
                100.0, 100.0, 100.0,
                0.0,  # Zero sales
            ],
        }
    )


def test_valid_purchase_lines_filters_non_positive_sales() -> None:
    # Given
    df = pd.DataFrame({
        "SALES_VALUE": [10.0, 0.0, -5.0, 25.5],
        "other_col": ["a", "b", "c", "d"],
    })

    # When
    filtered = valid_purchase_lines(df)

    # Then
    assert len(filtered) == 2
    assert list(filtered["SALES_VALUE"]) == [10.0, 25.5]


def test_valid_purchase_lines_raises_on_missing_column() -> None:
    # Given
    df = pd.DataFrame({"PRICE": [10.0, 20.0]})

    # When / Then
    with pytest.raises(PurchaseValidationError, match="Thiếu cột bắt buộc"):
        valid_purchase_lines(df)


def test_build_purchase_days_structure_and_gaps() -> None:
    # Given
    transactions = _synthetic_transactions()

    # When
    pdays = build_purchase_days(transactions)

    # Then
    assert not pdays.empty
    assert list(pdays.columns) == ["household_key", "DAY", "n_trips_day", "spend_day", "gap"]
    # HH 1 has 3 purchase days (10, 15, 20)
    hh1_days = pdays[pdays["household_key"] == 1].reset_index(drop=True)
    assert len(hh1_days) == 3
    assert hh1_days.loc[0, "DAY"] == 10
    assert hh1_days.loc[0, "n_trips_day"] == 2  # 2 trips on day 10 (baskets 101, 102)
    assert hh1_days.loc[0, "spend_day"] == 50.0  # 20.0 + 30.0
    assert pd.isna(hh1_days.loc[0, "gap"])
    assert hh1_days.loc[1, "DAY"] == 15
    assert hh1_days.loc[1, "gap"] == 5.0
    assert hh1_days.loc[2, "DAY"] == 20
    assert hh1_days.loc[2, "gap"] == 5.0


def test_build_household_summary_computes_expected_metrics() -> None:
    # Given
    transactions = _synthetic_transactions()

    # When
    summary = build_household_summary(transactions)

    # Then
    assert len(summary) == 4  # HH 5 is excluded
    assert list(summary["household_key"]) == [1, 2, 3, 4]

    # HH 1: 4 trips, 3 purchase days, spend=140, span=11 (day 10 to 20 -> 11 days)
    hh1 = summary.loc[summary["household_key"] == 1].iloc[0]
    assert hh1["first_day"] == 10
    assert hh1["last_day"] == 20
    assert hh1["n_purchase_days"] == 3
    assert hh1["n_trips"] == 4
    assert hh1["total_spend"] == 140.0
    assert hh1["span_days"] == 11
    assert bool(hh1["short_span"]) is True  # 11 < 28
    assert np.isclose(hh1["F"], 3 / (11 / 7.0))
    assert np.isclose(hh1["spend_per_week"], 140.0 / (11 / 7.0))
    assert np.isclose(hh1["spend_per_trip"], 140.0 / 4)
    # Gaps for HH 1: [15 - 10, 20 - 15] = [5, 5] -> median = 5, std = 0, cv = 0
    assert np.isclose(hh1["median_ipt"], 5.0)
    assert np.isclose(hh1["cv_ipt"], 0.0)

    # HH 2: 2 purchase days -> median_ipt is non-null, cv_ipt is null (< 3 days)
    hh2 = summary.loc[summary["household_key"] == 2].iloc[0]
    assert hh2["n_purchase_days"] == 2
    assert np.isclose(hh2["median_ipt"], 14.0)  # 24 - 10
    assert pd.isna(hh2["cv_ipt"])

    # HH 3: 1 purchase day -> median_ipt is null (< 2 days), cv_ipt is null
    hh3 = summary.loc[summary["household_key"] == 3].iloc[0]
    assert hh3["n_purchase_days"] == 1
    assert pd.isna(hh3["median_ipt"])
    assert pd.isna(hh3["cv_ipt"])

    # HH 4: span = 60 -> short_span is False
    hh4 = summary.loc[summary["household_key"] == 4].iloc[0]
    assert bool(hh4["short_span"]) is False


def test_build_household_summary_raises_on_missing_columns() -> None:
    # Given
    bad_df = pd.DataFrame({"household_key": [1], "DAY": [10]})

    # When / Then
    with pytest.raises(CustomerOverviewError, match="Thiếu các cột giao dịch bắt buộc"):
        build_household_summary(bad_df)


def test_describe_customer_metrics_generates_scorecard() -> None:
    # Given
    summary = build_household_summary(_synthetic_transactions())

    # When
    scorecard = describe_customer_metrics(summary)

    # Then
    assert not scorecard.empty
    assert "metric" in scorecard.columns
    assert "median" in scorecard.columns
    assert "n_null" in scorecard.columns

    # Verify cv_ipt reports 2 nulls (HH 2 and HH 3)
    cv_row = scorecard.loc[scorecard["metric"] == "cv_ipt"].iloc[0]
    assert cv_row["n_null"] == 2
    assert cv_row["n"] == 2


def test_gini_and_lorenz_curve() -> None:
    # 1. Perfect equality
    equal_vals = np.array([10.0, 10.0, 10.0, 10.0])
    assert np.isclose(gini(equal_vals), 0.0)

    p, L = lorenz_curve(equal_vals)
    assert len(p) == 5
    assert np.allclose(p, L)  # Lorenz curve equals 45-degree line

    # 2. Extreme inequality
    extreme_vals = np.array([0.0, 0.0, 0.0, 100.0])
    assert gini(extreme_vals) > 0.7

    # 3. Negative value raises error
    with pytest.raises(CustomerOverviewError, match="không âm"):
        gini([-5.0, 10.0])


def test_compute_concentration_stats() -> None:
    # Given
    summary = build_household_summary(_synthetic_transactions())

    # When
    stats = compute_concentration_stats(summary)

    # Then
    assert len(stats) == 2
    assert "gini_coefficient" in stats.columns
    assert "pct_households_for_50pct_total" in stats.columns


def test_plots_generate_valid_figures() -> None:
    # Given
    summary = build_household_summary(_synthetic_transactions())

    # When
    fig1 = plot_customer_overview_histograms(summary)
    fig2 = plot_customer_lorenz_curves(summary)

    # Then
    assert isinstance(fig1, Figure)
    assert len(fig1.axes) >= 5
    assert isinstance(fig2, Figure)

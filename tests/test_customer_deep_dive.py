"""Unit tests for customer deep-dive analytics: Q-F, Q-CV, Q-M, and Q-IPT."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from propensity.eda.customer_deep_dive import (
    CustomerDeepDiveError,
    CustomerDeepDiveReport,
    DeepDiveThresholds,
    SpendDecompositionResult,
    activity_continuity,
    build_customer_deep_dive_report,
    classify_high_f,
    cv_diagnostics,
    long_ipt_profile,
    spend_decomposition,
)
from propensity.eda.customer_deep_dive_plots import (
    plot_cv_diagnostics,
    plot_f_continuity,
    plot_ipt_band_profile,
    plot_spend_decomposition,
)
from propensity.eda.customer_overview import build_household_summary, build_purchase_days


def _synthetic_deep_dive_transactions() -> pd.DataFrame:
    """Generate synthetic transactions with clear behavioral profiles."""
    records = []
    # Household 1: Clockwork weekly shopper (high F, high coverage, span = 211d > 91d, max_share < 0.50)
    # Days: 0, 7, 14, ..., 210 -> 31 days, span = 211d, weekly coverage = 1.0, max_share = 41.9%
    for day in range(0, 217, 7):
        records.append({"household_key": 1, "BASKET_ID": 1000 + day, "DAY": day, "SALES_VALUE": 50.0})

    # Household 2: Burst shopper (high F within short span: 10 purchases in 20 days)
    # Days: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18 -> span = 19d (< 91d -> mua dồn)
    for day in range(0, 20, 2):
        records.append({"household_key": 2, "BASKET_ID": 2000 + day, "DAY": day, "SALES_VALUE": 40.0})

    # Household 3: Vacation break shopper (CV > 1 due to 1 long break, else regular)
    # 12 purchases: day 0, 7, 14, 21, 28, [vacation gap of 70 days], 98, 105, 112, 119, 126, 133
    days_hh3 = [0, 7, 14, 21, 28, 98, 105, 112, 119, 126, 133, 140]
    for idx, day in enumerate(days_hh3):
        records.append({"household_key": 3, "BASKET_ID": 3000 + idx, "DAY": day, "SALES_VALUE": 30.0})

    # Household 4: Infrequent long-IPT shopper (median IPT = 30 days, 4 purchases)
    days_hh4 = [10, 40, 70, 100]
    for idx, day in enumerate(days_hh4):
        records.append({"household_key": 4, "BASKET_ID": 4000 + idx, "DAY": day, "SALES_VALUE": 120.0})

    # Household 5: Low frequency, erratic shopper (n_gaps < 10)
    days_hh5 = [5, 15, 60]
    for idx, day in enumerate(days_hh5):
        records.append({"household_key": 5, "BASKET_ID": 5000 + idx, "DAY": day, "SALES_VALUE": 15.0})

    return pd.DataFrame(records)


def test_activity_continuity_computes_expected_metrics() -> None:
    # Given
    df = _synthetic_deep_dive_transactions()
    pdays = build_purchase_days(df)
    summary = build_household_summary(df)

    # When
    continuity = activity_continuity(pdays, summary)

    # Then
    assert len(continuity) == 5
    assert "weekly_coverage" in continuity.columns
    assert "max_window_share" in continuity.columns

    hh1 = continuity.loc[continuity["household_key"] == 1].iloc[0]
    # HH 1: purchased every 7 days from day 0 to 210 -> 31 distinct weeks
    assert hh1["n_purchase_days"] == 31
    assert hh1["weekly_coverage"] == 1.0


def test_classify_high_f_identifies_patterns() -> None:
    # Given
    df = _synthetic_deep_dive_transactions()
    pdays = build_purchase_days(df)
    summary = build_household_summary(df)
    continuity = activity_continuity(pdays, summary)

    # Set high_f_quantile lower so top households are classified
    thr = DeepDiveThresholds(high_f_quantile=0.50)

    # When
    high_f, summary_table = classify_high_f(continuity, thr)

    # Then
    assert not high_f.empty
    assert "f_pattern" in high_f.columns
    assert set(summary_table["f_pattern"]) == {"liên tục", "trung gian", "mua dồn"}

    # HH 2 has span 19 < 91 -> must be 'mua dồn'
    hh2_pattern = high_f.loc[high_f["household_key"] == 2, "f_pattern"].values[0]
    assert hh2_pattern == "mua dồn"

    # HH 1 has coverage 1.0 >= 0.75 -> must be 'liên tục'
    hh1_pattern = high_f.loc[high_f["household_key"] == 1, "f_pattern"].values[0]
    assert hh1_pattern == "liên tục"


def test_cv_diagnostics_detects_drivers() -> None:
    # Given
    df = _synthetic_deep_dive_transactions()
    pdays = build_purchase_days(df)
    summary = build_household_summary(df)

    # When
    diag, driver_sum = cv_diagnostics(pdays, summary)

    # Then
    assert len(diag) == 5
    assert "cv_driver" in diag.columns
    assert "cv_drop_max" in diag.columns

    # HH 1 has cv = 0.0 -> must be 'đều đặn'
    hh1 = diag.loc[diag["household_key"] == 1].iloc[0]
    assert hh1["cv_driver"] == "đều đặn"

    # HH 5 has n_gaps = 2 (< 10) -> if cv > 1, must be 'ít lần mua'
    hh5 = diag.loc[diag["household_key"] == 5].iloc[0]
    if hh5["cv_ipt"] > 1.0:
        assert hh5["cv_driver"] == "ít lần mua"

    # HH 3 has a single vacation gap of 70 days -> cv_drop_max drops significantly
    hh3 = diag.loc[diag["household_key"] == 3].iloc[0]
    assert hh3["n_long_breaks"] >= 1
    assert hh3["cv_drop_max"] < hh3["cv_ipt"]


def test_spend_decomposition_shares_sum_to_one() -> None:
    # Given
    df = _synthetic_deep_dive_transactions()
    summary = build_household_summary(df)

    # When (using exclude_short_span=False to include all test households)
    res = spend_decomposition(summary, exclude_short_span=False)

    # Then
    assert isinstance(res, SpendDecompositionResult)
    assert res.f_share + res.tpd_share + res.spt_share == pytest.approx(1.0, abs=1e-5)
    assert len(res.variance_shares) == 3
    assert "variance_share_pct" in res.variance_shares.columns
    assert len(res.decision_rule) > 10


def test_long_ipt_profile_and_subtypes() -> None:
    # Given
    df = _synthetic_deep_dive_transactions()
    pdays = build_purchase_days(df)
    summary = build_household_summary(df)
    continuity = activity_continuity(pdays, summary)
    diag, _ = cv_diagnostics(pdays, summary)

    # When
    bands_df, subtypes_df = long_ipt_profile(summary, continuity, diag)

    # Then
    assert not bands_df.empty
    assert "ipt_band" in bands_df.columns
    assert "pct_total_spend" in bands_df.columns

    # Subtypes table should cover the 5 defined subtypes
    assert not subtypes_df.empty
    assert set(subtypes_df["ipt_subtype"]) == {
        "lịch sử ngắn", "mua dự trữ", "mua bổ sung", "nhịp dài đều đặn", "khác"
    }


def test_build_customer_deep_dive_report_end_to_end() -> None:
    # Given
    df = _synthetic_deep_dive_transactions()

    # When
    report = build_customer_deep_dive_report(df)

    # Then
    assert isinstance(report, CustomerDeepDiveReport)
    assert len(report.purchase_days) > 0
    assert len(report.continuity) == 5
    assert isinstance(report.spend_decomposition, SpendDecompositionResult)


def test_all_deep_dive_plots_return_valid_figures() -> None:
    # Given
    df = _synthetic_deep_dive_transactions()
    report = build_customer_deep_dive_report(df)

    # When
    fig1 = plot_f_continuity(report, top_n=3)
    fig2 = plot_cv_diagnostics(report)
    fig3 = plot_spend_decomposition(report)
    fig4 = plot_ipt_band_profile(report)

    # Then
    assert isinstance(fig1, Figure)
    assert len(fig1.axes) >= 2
    assert isinstance(fig2, Figure)
    assert len(fig2.axes) >= 3
    assert isinstance(fig3, Figure)
    assert len(fig3.axes) >= 2
    assert isinstance(fig4, Figure)
    assert len(fig4.axes) >= 2

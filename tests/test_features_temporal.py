"""Unit tests for temporal feature extraction, trends, lapse bins, and cutoff stability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propensity.features.temporal import (
    cutoff_return_rates,
    lapse_bin,
    temporal_features,
    trend_bin,
    weekly_activity,
)


def test_temporal_features_ratio_and_minimum_gaps() -> None:
    """Verify recent_ipt_ratio_c requires >= 8 gaps (<9 distinct days gives NaN)."""
    # HH 1: 9 days (8 gaps)
    # Days: 10, 20, 30, 40, 50, 60, 70, 80, 90
    # Gaps: [10, 10, 10, 10, 10, 10, 10, 10]
    # last 3 gaps: [10, 10, 10] -> mean = 10
    # prior gaps: [10, 10, 10, 10, 10] -> median = 10
    # ratio = 1.0

    # HH 2: 8 days (7 gaps) -> must be NaN (<8 gaps)
    hh1_days = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    hh2_days = [10, 20, 30, 40, 50, 60, 70, 80]

    records = [{"household_key": 1, "DAY": d} for d in hh1_days] + [
        {"household_key": 2, "DAY": d} for d in hh2_days
    ]
    df = pd.DataFrame(records)

    res = temporal_features(df, cutoffs=[100])
    assert len(res) == 2

    row1 = res[res["household_key"] == 1].iloc[0]
    assert np.isclose(row1["recent_ipt_ratio_c"], 1.0)

    row2 = res[res["household_key"] == 2].iloc[0]
    assert pd.isna(row2["recent_ipt_ratio_c"])


def test_trend_bin() -> None:
    """Verify trend_bin correctly maps ratio into 4 categories."""
    s = pd.Series([0.5, 1.0, 1.5, 2.5, np.nan])
    binned = trend_bin(s)
    assert binned.iloc[0] == "< 0.75 (Tăng tốc)"
    assert binned.iloc[1] == "0.75–1.25 (Ổn định)"
    assert binned.iloc[2] == "1.25–2.0 (Chậm lại)"
    assert binned.iloc[3] == "> 2.0 (Chậm hẳn)"
    assert pd.isna(binned.iloc[4])


def test_lapse_bin() -> None:
    """Verify lapse_bin correctly maps lapse_ratio into 4 categories."""
    s = pd.Series([0.8, 1.5, 3.0, 5.0, np.nan])
    binned = lapse_bin(s)
    assert binned.iloc[0] == "≤ 1"
    assert binned.iloc[1] == "1–2"
    assert binned.iloc[2] == "2–4"
    assert binned.iloc[3] == "> 4"
    assert pd.isna(binned.iloc[4])


def test_weekly_activity() -> None:
    """Verify weekly_activity aggregates active households and baskets."""
    lines = pd.DataFrame(
        {
            "WEEK_NO": [1, 1, 1, 2],
            "household_key": [10, 10, 20, 30],
            "BASKET_ID": [1, 2, 3, 4],
            "DAY": [1, 2, 3, 8],
        }
    )

    act = weekly_activity(lines)
    assert len(act) == 2
    w1 = act[act["WEEK_NO"] == 1].iloc[0]
    assert w1["active_hh"] == 2
    assert w1["n_baskets"] == 3
    assert np.isclose(w1["baskets_per_active_hh"], 1.5)


def test_cutoff_return_rates_outlier_detection() -> None:
    """Verify cutoff_return_rates calculates rates and flags |robust_z| > 3.5."""
    # 20 cutoffs with rate varying slightly around 0.50, and 1 outlier with rate 0.95
    rows = []
    for c in range(1, 21):
        # Vary between 45 and 55 positives
        n_pos = 45 + (c % 11)
        for hh in range(100):
            rows.append({"cutoff_day": c, "household_key": hh, "target": 1 if hh < n_pos else 0})
    # Cutoff 99: extreme outlier (95 positives)
    for hh in range(100):
        rows.append({"cutoff_day": 99, "household_key": hh, "target": 1 if hh < 95 else 0})

    panel = pd.DataFrame(rows)
    rates = cutoff_return_rates(panel)
    assert len(rates) == 21

    normal_cutoffs = rates[rates["cutoff_day"] < 99]
    assert (normal_cutoffs["is_unusual"] == False).all()

    outlier_cutoff = rates[rates["cutoff_day"] == 99].iloc[0]
    assert outlier_cutoff["is_unusual"] == True
    assert outlier_cutoff["robust_z"] > 3.5

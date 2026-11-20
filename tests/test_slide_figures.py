"""Unit tests for slide_figures, slide_style, and feature_signal_summary."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from propensity.eda.propensity_rates import feature_signal_summary
from propensity.eda.slide_figures import (
    fig_c1,
    fig_c2,
    fig_d1,
    fig_l1,
    fig_l2a,
    fig_l2b,
    fig_l3,
    fig_p1,
    fig_p2,
    fig_q1,
    fig_v1,
    fig_v2,
    fig_v3,
    ipt_gap_distribution,
    table_coverage,
)
from propensity.viz.slide_style import set_slide_style


def test_ipt_gap_distribution_no_zero_and_order() -> None:
    """Check that ipt_gap_distribution produces no zero gaps and calculates correctly."""
    # Data with multiple purchases per household, some on same day (which should be collapsed or dropped)
    df = pd.DataFrame({
        "household_key": [1, 1, 1, 1, 2, 2, 3],
        "DAY": [10, 10, 13, 20, 5, 12, 100],
    })
    gaps = ipt_gap_distribution(df)

    # HH 1 distinct days: 10, 13, 20 -> gaps: 3, 7
    # HH 2 distinct days: 5, 12 -> gap: 7
    # HH 3 distinct days: 100 -> no gaps
    assert len(gaps) == 3
    assert (gaps >= 1).all()
    assert (gaps == 0).sum() == 0
    assert sorted(gaps.tolist()) == [3.0, 7.0, 7.0]

    # Cumulative calculations
    assert (gaps <= 3).mean() == pytest.approx(1 / 3)
    assert (gaps <= 7).mean() == pytest.approx(1.0)


def test_table_coverage() -> None:
    """Check table_coverage returns expected percentages and descending order."""
    tx = pd.DataFrame({"household_key": list(range(1, 2501))})
    camp = pd.DataFrame({"household_key": list(range(1, 1585))})
    demo = pd.DataFrame({"household_key": list(range(1, 802))})
    coup = pd.DataFrame({"household_key": list(range(1, 435))})

    cov = table_coverage(tx, camp, coup, demo, total_hh=2500)
    assert len(cov) == 4
    assert cov.iloc[0]["pct_households"] == 100.0
    assert cov.iloc[1]["pct_households"] == pytest.approx(63.36, abs=0.1)
    assert cov.iloc[2]["pct_households"] == pytest.approx(32.04, abs=0.1)
    assert cov.iloc[3]["pct_households"] == pytest.approx(17.36, abs=0.1)
    # Check descending order
    assert cov["pct_households"].is_monotonic_decreasing


def test_feature_signal_summary_drops_null_and_insufficient() -> None:
    """Check that feature_signal_summary excludes null bins and 'Không đủ dữ liệu' bins."""
    res = {
        "Feature A": pd.DataFrame({
            "group": ["Q1", "Q2", "null", "Q3"],
            "raw_diff_pp": [-10.0, 5.0, -35.0, 15.0],
            "excess_pp": [-1.0, 0.5, -7.0, 1.2],
            "verdict": ["Đã được R × F giải thích", "Đã được R × F giải thích", "Insight", "Đã được R × F giải thích"],
        }),
        "Feature B": pd.DataFrame({
            "band": ["Low", "High", "Unknown"],
            "raw_diff_pp": [5.0, 25.0, 30.0],
            "excess_pp": [0.5, 4.0, 6.0],
            "verdict": ["Không được ủng hộ", "Insight", "Không đủ dữ liệu"],
        }),
    }

    summary = feature_signal_summary(res)
    assert len(summary) == 2

    # Feature A: null is dropped -> max |raw_diff_pp| is 15.0 (not 35.0), max |excess_pp| is 1.2, verdict is Đã được R × F giải thích
    row_a = summary[summary["feature"] == "Feature A"].iloc[0]
    assert row_a["raw_pp"] == 15.0
    assert row_a["adj_pp"] == 1.2
    assert row_a["best_verdict"] == "Đã được R × F giải thích"

    # Feature B: 'Không đủ dữ liệu' dropped -> max |raw_diff_pp| is 25.0, max |excess_pp| is 4.0, verdict is Insight
    row_b = summary[summary["feature"] == "Feature B"].iloc[0]
    assert row_b["raw_pp"] == 25.0
    assert row_b["adj_pp"] == 4.0
    assert row_b["best_verdict"] == "Insight"

    # Sorted descending by raw_pp -> Feature B (25.0) comes before Feature A (15.0)
    assert summary.iloc[0]["feature"] == "Feature B"


@pytest.mark.parametrize(
    "fig_func, sample_data",
    [
        (
            "fig_v1",
            pd.DataFrame({
                "r_bin": ["1–7"] * 5 + ["8–14"] * 5 + ["15–30"] * 5 + ["31+"] * 5,
                "f_bin": ["≤ 0.30", "0.30–0.57", "0.57–0.96", "0.96–1.52", "> 1.52"] * 4,
                "rate": np.linspace(0.1, 0.9, 20),
            }),
        ),
        (
            "fig_v2",
            pd.DataFrame({
                "feature": ["F1", "F2", "F3"],
                "raw_pp": [30.0, 20.0, 10.0],
                "adj_pp": [1.5, 7.3, 0.5],
                "best_verdict": ["Đã được R × F giải thích", "Insight", "Không được ủng hộ"],
            }),
        ),
        (
            "fig_v3",
            pd.DataFrame({
                "trend_bin": ["< 0.75 (Tăng tốc)", "0.75–1.25 (Ổn định)", "1.25–2.0 (Chậm lại)", "> 2.0 (Chậm hẳn)"],
                "observed": [0.645, 0.683, 0.625, 0.439],
                "expected": [0.575, 0.646, 0.621, 0.511],
                "excess_pp": [7.0, 3.7, 0.4, -7.3],
                "verdict": ["Insight", "Insight", "Đã được R × F giải thích", "Insight"],
                "n_households": [800, 1100, 950, 400],
            }),
        ),
        (
            "fig_l1",
            pd.DataFrame({
                "WEEK_NO": list(range(1, 103)),
                "active_hh": [1400] * 20 + [2250] * 82,
                "min_day": [i * 7 for i in range(1, 103)],
            }),
        ),
        (
            "fig_l2a",
            pd.Series(np.random.choice(range(1, 40), size=500)),
        ),
        (
            "fig_l2b",
            pd.Series(np.random.choice(range(1, 40), size=500)),
        ),
        (
            "fig_l3",
            pd.DataFrame({
                "cutoff_day": np.arange(140, 701, 7),
                "return_rate": np.random.uniform(0.50, 0.56, size=81),
            }),
        ),
        (
            "fig_c1",
            pd.DataFrame({
                "band": ["1–2", "3–5", "6–10", "11–20", "21+"],
                "pct_baskets": [0.274, 0.233, 0.200, 0.152, 0.141],
                "pct_spend": [0.055, 0.096, 0.150, 0.225, 0.474],
                "median_spend": [3.8, 10.1, 22.5, 45.0, 87.1],
            }),
        ),
        (
            "fig_c2",
            pd.DataFrame({
                "DEPARTMENT": ["GROCERY", "DRUG GM", "PRODUCE", "MEAT", "DELI", "PASTRY", "SEAFOOD", "NUTRITION", "SALAD", "FLORAL"],
                "spend": [3000000, 1500000, 800000, 600000, 300000, 200000, 100000, 80000, 60000, 40000],
                "spend_share": [0.555, 0.150, 0.080, 0.062, 0.035, 0.025, 0.015, 0.010, 0.008, 0.005],
                "cum_spend_share": [0.555, 0.705, 0.785, 0.847, 0.882, 0.907, 0.922, 0.932, 0.940, 0.945],
            }),
        ),
        (
            "fig_p1",
            pd.DataFrame({
                "WEEK_NO": list(range(1, 103)),
                "promo_spend_share": np.random.uniform(0.40, 0.55, size=102),
            }),
        ),
        (
            "fig_p2",
            pd.DataFrame({
                "F_bin": ["≤ 0.30", "0.30–0.57", "0.57–0.96", "0.96–1.52", "> 1.52"],
                "reach_rate": [0.0023, 0.05, 0.15, 0.25, 0.389],
            }),
        ),
        (
            "fig_d1",
            pd.DataFrame({
                "table_name": ["Giao dịch", "Chiến dịch", "Nhân khẩu học", "Coupon đã đổi"],
                "n_households": [2500, 1584, 801, 434],
                "pct_households": [100.0, 63.4, 32.0, 17.4],
            }),
        ),
        (
            "fig_q1",
            pd.DataFrame([
                {
                    "group": "Có demographic",
                    "has_demographic": 1,
                    "f_≤ 0.30": 0.030,
                    "f_0.30–0.57": 0.076,
                    "f_0.57–0.96": 0.190,
                    "f_0.96–1.52": 0.322,
                    "f_> 1.52": 0.381,
                },
                {
                    "group": "Không có demographic",
                    "has_demographic": 0,
                    "f_≤ 0.30": 0.280,
                    "f_0.30–0.57": 0.258,
                    "f_0.57–0.96": 0.204,
                    "f_0.96–1.52": 0.142,
                    "f_> 1.52": 0.114,
                },
            ]),
        ),
    ],
)
def test_all_figures_single_axis_no_suptitle(fig_func: str, sample_data: Any) -> None:
    """Verify that every slide figure function generates exactly 1 axis and NO suptitle."""
    from propensity.eda import slide_figures

    fn = getattr(slide_figures, fig_func)
    fig = fn(sample_data)

    assert isinstance(fig, plt.Figure)
    # Exactly 1 axis (or 2 if colorbar added an axis for imshow in fig_v1)
    if fig_func == "fig_v1":
        # fig_v1 has main ax and colorbar ax
        assert len(fig.axes) in (1, 2)
    else:
        assert len(fig.axes) == 1

    # No suptitle
    assert fig._suptitle is None or fig._suptitle.get_text() == ""
    plt.close(fig)

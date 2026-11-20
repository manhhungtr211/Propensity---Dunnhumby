"""Unit tests for propensity_rates: asof_aggregate, join_features, rf_baseline, excess_over_rf, and verdicts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propensity.eda.customer_label_relation import (
    CustomerLabelConfig,
    build_customer_cutoff_panel,
)
from propensity.eda.propensity_rates import (
    FindingThresholds,
    RFBaseline,
    add_verdicts,
    asof_aggregate,
    classify_verdict,
    excess_over_rf,
    join_features,
    rf_baseline,
)


def test_asof_aggregate_strict_point_in_time() -> None:
    """Test that asof_aggregate strictly excludes DAY >= cutoff."""
    data = pd.DataFrame(
        {
            "household_key": [1, 1, 1, 2, 2],
            "DAY": [10, 20, 30, 15, 25],
            "val": [100, 200, 300, 50, 60],
        }
    )

    def agg_fn(sub: pd.DataFrame) -> pd.DataFrame:
        if sub.empty:
            return pd.DataFrame(columns=["household_key", "sum_val", "max_day"])
        return (
            sub.groupby("household_key", as_index=False)
            .agg(sum_val=("val", "sum"), max_day=("DAY", "max"))
        )

    res = asof_aggregate(data, cutoffs=[25, 35], agg_fn=agg_fn)

    # At cutoff 25: HH 1 has days 10, 20 (day 30 excluded!); HH 2 has day 15 (day 25 excluded!)
    c25 = res[res["cutoff_day"] == 25].set_index("household_key")
    assert c25.loc[1, "sum_val"] == 300
    assert c25.loc[1, "max_day"] == 20
    assert c25.loc[2, "sum_val"] == 50
    assert c25.loc[2, "max_day"] == 15

    # At cutoff 35: HH 1 has days 10, 20, 30 (sum 600); HH 2 has days 15, 25 (sum 110)
    c35 = res[res["cutoff_day"] == 35].set_index("household_key")
    assert c35.loc[1, "sum_val"] == 600
    assert c35.loc[1, "max_day"] == 30
    assert c35.loc[2, "sum_val"] == 110


def test_asof_aggregate_duplicate_keys_raise() -> None:
    """Test that agg_fn producing duplicate keys raises ValueError."""
    data = pd.DataFrame({"household_key": [1, 1], "DAY": [10, 20]})

    def bad_agg(sub: pd.DataFrame) -> pd.DataFrame:
        # returns duplicate key
        return pd.DataFrame({"household_key": [1, 1], "feat": [1, 2]})

    with pytest.raises(ValueError, match="bị trùng"):
        asof_aggregate(data, cutoffs=[30], agg_fn=bad_agg)


def test_join_features_validations() -> None:
    """Test join_features row count preservation, duplicate key error, and column collision error."""
    panel = pd.DataFrame(
        {
            "household_key": [1, 2, 1, 2],
            "cutoff_day": [140, 140, 147, 147],
            "target": [1, 0, 0, 1],
        }
    )

    frame1 = pd.DataFrame(
        {
            "household_key": [1, 2, 1, 2],
            "cutoff_day": [140, 140, 147, 147],
            "feat_a": [10.0, 20.0, 30.0, 40.0],
        }
    )

    # 1. Normal successful join
    joined = join_features(panel, frame1)
    assert len(joined) == len(panel)
    assert "feat_a" in joined.columns

    # 2. Duplicate keys in frame raises
    bad_dup_frame = pd.DataFrame(
        {
            "household_key": [1, 1],
            "cutoff_day": [140, 140],
            "feat_b": [1, 2],
        }
    )
    with pytest.raises(ValueError, match="trùng lặp"):
        join_features(panel, bad_dup_frame)

    # 3. Column collision raises
    colliding_frame = pd.DataFrame(
        {
            "household_key": [1, 2],
            "cutoff_day": [140, 140],
            "target": [99, 99],
        }
    )
    with pytest.raises(ValueError, match="Xung đột tên cột"):
        join_features(panel, colliding_frame)


def test_rf_baseline_cell_and_margin_fallback() -> None:
    """Test rf_baseline 2x2 grid calculation and unreliable cell fallback to r_margin."""
    # Create synthetic panel:
    # R1 has R1-F1 (60 households -> reliable) and R1-F2 (20 households -> unreliable)
    # R2 has R2-F1 (100 households -> reliable)
    rows = []
    # R1-F1: 60 households, all target=1 -> rate = 1.0
    for h in range(1, 61):
        rows.append({"household_key": h, "recency_bin": "R1", "F_bin": "F1", "target": 1, "cutoff_day": 140})
    # R1-F2: 20 households (<50), all target=0 -> cell rate = 0.0, but R1 margin rate = 60/80 = 0.75
    for h in range(61, 81):
        rows.append({"household_key": h, "recency_bin": "R1", "F_bin": "F2", "target": 0, "cutoff_day": 140})
    # R2-F1: 100 households, 40 positive -> rate = 0.40
    for h in range(81, 181):
        target = 1 if h <= 120 else 0
        rows.append({"household_key": h, "recency_bin": "R2", "F_bin": "F1", "target": target, "cutoff_day": 140})

    panel = pd.DataFrame(rows)
    rf = rf_baseline(panel, min_households=50)

    # Check cells
    c_r1_f1 = rf.cells[(rf.cells["r_bin"] == "R1") & (rf.cells["f_bin"] == "F1")].iloc[0]
    assert c_r1_f1["is_reliable"] is True or c_r1_f1["is_reliable"] == 1
    assert c_r1_f1["rate"] == 1.0

    c_r1_f2 = rf.cells[(rf.cells["r_bin"] == "R1") & (rf.cells["f_bin"] == "F2")].iloc[0]
    assert c_r1_f2["is_reliable"] is False or c_r1_f2["is_reliable"] == 0

    # Check panel mapping
    p_r1_f1 = rf.panel[(rf.panel["recency_bin"] == "R1") & (rf.panel["F_bin"] == "F1")]
    assert (p_r1_f1["expected_source"] == "cell").all()
    assert (p_r1_f1["expected_rf"] == 1.0).all()

    p_r1_f2 = rf.panel[(rf.panel["recency_bin"] == "R1") & (rf.panel["F_bin"] == "F2")]
    assert (p_r1_f2["expected_source"] == "r_margin").all()
    # R1 margin rate is 60 / 80 = 0.75
    assert np.isclose(p_r1_f2["expected_rf"].iloc[0], 0.75)


def test_excess_over_rf_calculations() -> None:
    """Test excess_pp, raw_diff_pp, and same_sign_share across cutoffs."""
    # Synthetic panel with 3 cutoffs (140, 147, 154)
    # Group G1: observed 1.0, expected 0.5 -> excess_pp = +50.0 pp
    # Across all 3 cutoffs, observed (1.0) > expected (0.5), so same_sign_share = 1.0
    rows = []
    for c in [140, 147, 154]:
        for h in range(1, 41):  # 40 rows per cutoff >= 20 min_rows_per_cutoff
            rows.append(
                {
                    "household_key": h,
                    "cutoff_day": c,
                    "my_group": "G1",
                    "target": 1,
                    "expected_rf": 0.5,
                }
            )
        for h in range(41, 81):
            rows.append(
                {
                    "household_key": h,
                    "cutoff_day": c,
                    "my_group": "G2",
                    "target": 0,
                    "expected_rf": 0.0,  # match cell -> excess_pp = 0
                }
            )

    panel = pd.DataFrame(rows)
    excess_df = excess_over_rf(panel, group_col="my_group")

    g1_row = excess_df[excess_df["my_group"] == "G1"].iloc[0]
    assert np.isclose(g1_row["observed"], 1.0)
    assert np.isclose(g1_row["expected"], 0.5)
    assert np.isclose(g1_row["excess_pp"], 50.0)
    assert g1_row["same_sign_share"] == 1.0
    assert g1_row["n_cutoffs_used"] == 3

    g2_row = excess_df[excess_df["my_group"] == "G2"].iloc[0]
    assert np.isclose(g2_row["observed"], 0.0)
    assert np.isclose(g2_row["expected"], 0.0)
    assert np.isclose(g2_row["excess_pp"], 0.0)


def test_finding_thresholds_defaults() -> None:
    """Verify default finding thresholds including min_row_share=0.02."""
    thr = FindingThresholds()
    assert thr.min_effect_pp == 3.0
    assert thr.min_same_sign == 0.80
    assert thr.min_row_share == 0.02
    assert thr.min_households == 50
    assert thr.min_rows_per_cutoff == 20


def test_classify_verdict_branches() -> None:
    """Verify all 5 decision branches of classify_verdict."""
    thr = FindingThresholds(
        min_effect_pp=3.0,
        min_same_sign=0.80,
        min_row_share=0.05,
        min_households=50,
    )

    # Branch 1: Không đủ dữ liệu (too few households or small share)
    r1 = {"row_share": 0.02, "n_households": 20, "raw_diff_pp": 10.0, "excess_pp": 8.0, "same_sign_share": 0.9}
    v1, _ = classify_verdict(r1, thr)
    assert v1 == "Không đủ dữ liệu"

    # Branch 2: Đã được R × F giải thích (|raw_diff| >= 3, |excess| < 3)
    r2 = {"row_share": 0.10, "n_households": 100, "raw_diff_pp": 8.0, "excess_pp": 1.2, "same_sign_share": 0.9}
    v2, _ = classify_verdict(r2, thr)
    assert v2 == "Đã được R × F giải thích"

    # Branch 3: Không được ủng hộ (|excess| < 3 and |raw_diff| < 3)
    r3 = {"row_share": 0.10, "n_households": 100, "raw_diff_pp": 1.5, "excess_pp": 1.1, "same_sign_share": 0.9}
    v3, _ = classify_verdict(r3, thr)
    assert v3 == "Không được ủng hộ"

    # Branch 4: Không ổn định (same_sign_share < 0.8)
    r4 = {"row_share": 0.10, "n_households": 100, "raw_diff_pp": 10.0, "excess_pp": 6.5, "same_sign_share": 0.6}
    v4, _ = classify_verdict(r4, thr)
    assert v4 == "Không ổn định"

    # Branch 5: Insight (supported and stable)
    r5 = {"row_share": 0.10, "n_households": 100, "raw_diff_pp": 10.0, "excess_pp": 6.5, "same_sign_share": 0.85}
    v5, _ = classify_verdict(r5, thr)
    assert v5 == "Insight"

    # Test add_verdicts table
    df = pd.DataFrame([r1, r2, r3, r4, r5])
    df_verdicts = add_verdicts(df, thr)
    assert list(df_verdicts["verdict"]) == [
        "Không đủ dữ liệu",
        "Đã được R × F giải thích",
        "Không được ủng hộ",
        "Không ổn định",
        "Insight",
    ]


def test_build_customer_cutoff_panel_lapse_ratio() -> None:
    """Verify lapse_ratio_c is computed as recency / median_ipt and null for 1 purchase day."""
    cfg = CustomerLabelConfig(first_cutoff_day=140, last_cutoff_day=140)
    # HH 1: days 10, 88 -> gaps = [78] -> median_ipt = 78. At c=140, recency = 140 - 88 = 52.
    # lapse_ratio_c = 52 / 78
    # HH 2: only day 20 -> 1 purchase day -> median_ipt is NaN -> lapse_ratio_c is NaN
    records = [
        {"household_key": 1, "BASKET_ID": 1, "DAY": 10, "SALES_VALUE": 20.0, "QUANTITY": 1},
        {"household_key": 1, "BASKET_ID": 2, "DAY": 88, "SALES_VALUE": 30.0, "QUANTITY": 1},
        {"household_key": 2, "BASKET_ID": 3, "DAY": 20, "SALES_VALUE": 15.0, "QUANTITY": 1},
    ]
    panel = build_customer_cutoff_panel(pd.DataFrame(records), cfg)

    assert "lapse_ratio_c" in panel.columns

    row1 = panel[panel["household_key"] == 1].iloc[0]
    assert np.isclose(row1["lapse_ratio_c"], 52.0 / 78.0)

    row2 = panel[panel["household_key"] == 2].iloc[0]
    assert pd.isna(row2["lapse_ratio_c"])

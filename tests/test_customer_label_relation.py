"""Unit tests for customer-to-label relation analytics and leakage prevention."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propensity.eda.customer_label_relation import (
    CustomerLabelConfig,
    CustomerLabelRelationError,
    build_customer_cutoff_panel,
    classify_finding,
    compare_purchase_rules,
    excess_over_baseline,
    rate_within_strata,
    return_rate_by_bins,
)


def _make_synthetic_transactions() -> pd.DataFrame:
    """Create deterministic synthetic transaction history for leakage and logic tests."""
    records = []
    # Household 1: Buys regularly before cutoff 140, and buys on day 142 (Target=1 at c=140)
    for day, basket in [(10, 101), (30, 102), (60, 103), (90, 104), (120, 105), (142, 106)]:
        records.append(
            {
                "household_key": 1,
                "BASKET_ID": basket,
                "DAY": day,
                "SALES_VALUE": 25.0,
                "QUANTITY": 2,
            }
        )

    # Household 2: Buys only before cutoff 140, no purchase in [140, 146] (Target=0 at c=140)
    for day, basket in [(15, 201), (45, 202), (75, 203), (105, 204), (160, 205)]:
        records.append(
            {
                "household_key": 2,
                "BASKET_ID": basket,
                "DAY": day,
                "SALES_VALUE": 40.0,
                "QUANTITY": 1,
            }
        )

    # Household 3: Brand new household at day 145 (not eligible at c=140)
    records.append(
        {
            "household_key": 3,
            "BASKET_ID": 301,
            "DAY": 145,
            "SALES_VALUE": 15.0,
            "QUANTITY": 1,
        }
    )

    # Household 4: Has a zero-price promotion line (SALES_VALUE=0) on day 50
    records.append(
        {
            "household_key": 4,
            "BASKET_ID": 401,
            "DAY": 50,
            "SALES_VALUE": 0.0,
            "QUANTITY": 1,
        }
    )
    records.append(
        {
            "household_key": 4,
            "BASKET_ID": 402,
            "DAY": 70,
            "SALES_VALUE": 30.0,
            "QUANTITY": 1,
        }
    )

    return pd.DataFrame(records)


def test_strict_temporal_leakage_discipline() -> None:
    """Test that modifying transactions on or after cutoff does not alter pre-cutoff features."""
    cfg = CustomerLabelConfig(
        first_cutoff_day=140,
        last_cutoff_day=140,
        cutoff_step_days=7,
        horizon_days=7,
    )
    df_base = _make_synthetic_transactions()
    panel_base = build_customer_cutoff_panel(df_base, cfg)

    row_hh1_base = panel_base.loc[panel_base["household_key"] == 1].iloc[0]
    assert row_hh1_base["n_pdays"] == 5  # days 10, 30, 60, 90, 120
    assert row_hh1_base["recency"] == 20  # 140 - 120
    assert row_hh1_base["target"] == 1  # bought on day 142

    # Now add massive transactions after cutoff 140 (e.g. DAY 141, 150, 200)
    df_future = df_base.copy()
    leak_rows = pd.DataFrame(
        [
            {"household_key": 1, "BASKET_ID": 999, "DAY": 141, "SALES_VALUE": 9999.0, "QUANTITY": 50},
            {"household_key": 1, "BASKET_ID": 1000, "DAY": 180, "SALES_VALUE": 5000.0, "QUANTITY": 20},
        ]
    )
    df_future = pd.concat([df_future, leak_rows], ignore_index=True)
    panel_future = build_customer_cutoff_panel(df_future, cfg)

    row_hh1_future = panel_future.loc[panel_future["household_key"] == 1].iloc[0]
    # Features MUST be 100% identical!
    assert row_hh1_future["n_pdays"] == row_hh1_base["n_pdays"]
    assert row_hh1_future["recency"] == row_hh1_base["recency"]
    assert row_hh1_future["tenure"] == row_hh1_base["tenure"]
    assert row_hh1_future["F_c"] == row_hh1_base["F_c"]
    assert row_hh1_future["spend_per_trip"] == row_hh1_base["spend_per_trip"]
    assert row_hh1_future["median_ipt"] == row_hh1_base["median_ipt"]


def test_eligibility_and_target_assignment() -> None:
    """Verify eligibility at cutoff c and target window [c, c + 6]."""
    cfg = CustomerLabelConfig(
        first_cutoff_day=140,
        last_cutoff_day=140,
        cutoff_step_days=7,
        horizon_days=7,
    )
    df = _make_synthetic_transactions()
    panel = build_customer_cutoff_panel(df, cfg)

    # HH 3 first purchased on day 145 -> NOT eligible at c=140
    assert 3 not in panel["household_key"].values

    # HH 1 bought on day 142 -> target = 1
    assert panel.loc[panel["household_key"] == 1, "target"].iloc[0] == 1

    # HH 2 bought on day 160 (outside [140, 146]) -> target = 0
    assert panel.loc[panel["household_key"] == 2, "target"].iloc[0] == 0


def test_return_rate_and_strata_helpers() -> None:
    """Test return rate by bin, Wilson CI, and strata cross-tabulation."""
    # Synthetic panel
    panel = pd.DataFrame(
        {
            "F_bin": ["Q1", "Q1", "Q2", "Q2", "Q3", "Q3"],
            "recency_bin": ["1–7", "8–14", "1–7", "8–14", "1–7", "8–14"],
            "spend_bin": ["Q1", "Q1", "Q2", "Q2", "Q3", "Q3"],
            "target": [0, 1, 1, 1, 1, 0],
        }
    )

    rates = return_rate_by_bins(panel, "F_bin")
    assert len(rates) == 3
    assert set(rates.columns).issuperset({"F_bin", "n", "n_pos", "return_rate", "ci_lower", "ci_upper"})
    assert (rates["ci_lower"] <= rates["return_rate"]).all()
    assert (rates["return_rate"] <= rates["ci_upper"]).all()

    strata = rate_within_strata(panel, "F_bin", "recency_bin")
    assert strata.shape == (3, 2)
    assert strata.loc["Q2", "1–7"] == 1.0

    excess = excess_over_baseline(panel, "spend_bin")
    assert "excess_over_baseline" in excess.columns
    assert "excess_over_rf" in excess.columns
    assert np.isclose(excess["excess_over_baseline"].mean(), 0.0, atol=1e-5)


def test_classify_finding_and_purchase_rules() -> None:
    """Test generic threshold-based classification helper and compare_purchase_rules."""
    # 1. Test core feature (large spread >= 15 pp)
    df_core = pd.DataFrame({"excess_over_rf": [-0.10, 0.15]})
    c_core = classify_finding(df_core)
    assert c_core["is_core"]
    assert "Core" in c_core["classification"]
    assert c_core["spread_pp"] == 25.0

    # 2. Test secondary feature (moderate spread 3-15 pp)
    df_sec = pd.DataFrame({"excess_over_rf": [-0.02, 0.03]})
    c_sec = classify_finding(df_sec)
    assert c_sec["is_secondary"]
    assert "Secondary" in c_sec["classification"]

    # 3. Test redundant / disguised feature with strata matrix
    strata_flat = pd.DataFrame({"1-7": [0.50, 0.505], "8-14": [0.30, 0.302]})
    c_flat = classify_finding(df_core, strata_matrix=strata_flat)
    assert c_flat["is_redundant"]
    assert "Redundant" in c_flat["classification"]

    df_trans = _make_synthetic_transactions()
    cfg = CustomerLabelConfig(first_cutoff_day=140, last_cutoff_day=140)
    rule_diff = compare_purchase_rules(df_trans, cfg)
    assert rule_diff["n_zero_price_lines"] == 1
    assert "Outlier / Data Quality" in rule_diff["classification"]

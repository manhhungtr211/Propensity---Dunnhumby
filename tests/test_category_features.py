"""Unit tests for category features, staple definition, and department indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propensity.eda.propensity_rates import quantile_bin
from propensity.features.category import (
    CategoryConfig,
    category_features,
    commodity_stats,
    define_staples,
    department_flags,
    department_summary,
    evaluate_department_flags_excess,
    excluded_department_summary,
    household_day_category,
    merch_lines,
    select_flag_departments,
)


def _make_synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic transactions and product catalog for deterministic tests."""
    tx = pd.DataFrame(
        [
            # HH 1: buys Milk (Grocery, Private) and Gas on day 50, and Produce on day 100
            {"household_key": 1, "BASKET_ID": 101, "DAY": 50, "PRODUCT_ID": 1, "SALES_VALUE": 10.0, "QUANTITY": 2},
            {"household_key": 1, "BASKET_ID": 101, "DAY": 50, "PRODUCT_ID": 2, "SALES_VALUE": 30.0, "QUANTITY": 1},  # Gas
            {"household_key": 1, "BASKET_ID": 102, "DAY": 100, "PRODUCT_ID": 3, "SALES_VALUE": 15.0, "QUANTITY": 3},
            # HH 2: buys Milk (day 20 and day 60), Produce (day 150 - outside window 139)
            {"household_key": 2, "BASKET_ID": 201, "DAY": 20, "PRODUCT_ID": 1, "SALES_VALUE": 8.0, "QUANTITY": 1},
            {"household_key": 2, "BASKET_ID": 202, "DAY": 60, "PRODUCT_ID": 1, "SALES_VALUE": 8.0, "QUANTITY": 1},
            {"household_key": 2, "BASKET_ID": 203, "DAY": 150, "PRODUCT_ID": 3, "SALES_VALUE": 12.0, "QUANTITY": 2},
            # HH 3: buys an unmatched product ID (999)
            {"household_key": 3, "BASKET_ID": 301, "DAY": 40, "PRODUCT_ID": 999, "SALES_VALUE": 5.0, "QUANTITY": 1},
        ]
    )

    product = pd.DataFrame(
        [
            {"PRODUCT_ID": 1, "DEPARTMENT": "GROCERY", "COMMODITY_DESC": "MILK", "BRAND": "Private"},
            {"PRODUCT_ID": 2, "DEPARTMENT": "KIOSK-GAS", "COMMODITY_DESC": "GASOLINE-REG UNLEADED", "BRAND": "National"},
            {"PRODUCT_ID": 3, "DEPARTMENT": "PRODUCE", "COMMODITY_DESC": "APPLES", "BRAND": "National"},
        ]
    )
    return tx, product


def test_merch_lines_enrichment_and_flags() -> None:
    """Verify merch_lines flags merchandise, fuel, private label, and handles unmatched products."""
    tx, product = _make_synthetic_data()
    cfg = CategoryConfig()
    lines = merch_lines(tx, product, cfg)

    assert len(lines) == len(tx)
    row_milk = lines[lines["PRODUCT_ID"] == 1].iloc[0]
    assert row_milk["is_merch"] is True or row_milk["is_merch"] == 1
    assert row_milk["is_fuel"] is False or row_milk["is_fuel"] == 0
    assert row_milk["is_private"] is True or row_milk["is_private"] == 1

    row_gas = lines[lines["PRODUCT_ID"] == 2].iloc[0]
    assert row_gas["is_fuel"] is True or row_gas["is_fuel"] == 1
    assert row_gas["is_merch"] is False or row_gas["is_merch"] == 0

    # Unmatched product 999 should not be merchandise
    row_unmatched = lines[lines["PRODUCT_ID"] == 999].iloc[0]
    assert row_unmatched["is_merch"] is False or row_unmatched["is_merch"] == 0


def test_excluded_and_department_summaries() -> None:
    """Test excluded_department_summary and department_summary metrics."""
    tx, product = _make_synthetic_data()
    cfg = CategoryConfig()
    lines = merch_lines(tx, product, cfg)

    # Excluded summary
    ex = excluded_department_summary(lines)
    assert not ex.empty
    gas_row = ex[ex["department"] == "KIOSK-GAS"].iloc[0]
    assert gas_row["spend"] == 30.0

    # Merch department summary
    dept = department_summary(lines, cfg)
    assert "cum_spend_share" in dept.columns
    assert "penetration_early" in dept.columns
    assert len(dept) == 2  # GROCERY and PRODUCE


def test_commodity_stats_and_define_staples() -> None:
    """Verify commodity_stats calculation strictly on DAY <= staple_max_day and define_staples."""
    tx, product = _make_synthetic_data()
    cfg = CategoryConfig(min_buyers=2, min_penetration=0.5, max_median_gap=50.0)
    lines = merch_lines(tx, product, cfg)

    stats = commodity_stats(lines, cfg)
    assert not stats.empty
    # HH 1 bought Milk (day 50); HH 2 bought Milk (day 20, day 60) -> 2 buyers
    milk_row = stats[stats["commodity"] == "MILK"].iloc[0]
    assert milk_row["n_buyers"] == 2
    assert milk_row["repeat_buyer_share"] == 0.5  # HH 2 is repeat buyer
    assert milk_row["median_gap"] == 40.0  # 60 - 20 = 40

    staples = define_staples(stats, cfg)
    milk_staple = staples[staples["commodity"] == "MILK"].iloc[0]
    assert milk_staple["is_staple"] is True or milk_staple["is_staple"] == 1
    assert milk_staple["staple_reason"] == "Staple"


def test_category_features_point_in_time() -> None:
    """Test category_features strict point-in-time calculation across cutoffs."""
    tx, product = _make_synthetic_data()
    lines = merch_lines(tx, product)
    staple_set = {"MILK"}
    hdc = household_day_category(lines, staple_set)

    # At cutoff 70:
    # HH 1 has day 50: Milk spend = 10, Gas spend = 30. Merch spend = 10, Staple spend = 10, Private = 10.
    # staple_spend_share_c = 1.0, private_label_share_c = 1.0, fuel_share_c = 30 / 40 = 0.75, buys_fuel = 1.
    # dept_entropy_c = 0.0 (only GROCERY), n_departments_c = 1.
    feats = category_features(hdc, cutoffs=[70, 120])

    hh1_c70 = feats[(feats["household_key"] == 1) & (feats["cutoff_day"] == 70)].iloc[0]
    assert np.isclose(hh1_c70["staple_spend_share_c"], 1.0)
    assert np.isclose(hh1_c70["private_label_share_c"], 1.0)
    assert np.isclose(hh1_c70["fuel_share_c"], 30.0 / 40.0)
    assert hh1_c70["buys_fuel_c"] == 1
    assert hh1_c70["n_departments_c"] == 1
    assert np.isclose(hh1_c70["dept_entropy_c"], 0.0)

    # At cutoff 120:
    # HH 1 adds Produce (day 100, spend 15).
    # Merch spend = 25 (10 Grocery, 15 Produce). Entropy > 0, n_departments_c = 2.
    hh1_c120 = feats[(feats["household_key"] == 1) & (feats["cutoff_day"] == 120)].iloc[0]
    assert hh1_c120["n_departments_c"] == 2
    assert hh1_c120["dept_entropy_c"] > 0.0
    assert np.isclose(hh1_c120["staple_spend_share_c"], 10.0 / 25.0)


def test_department_flags_point_in_time() -> None:
    """Test department_flags indicator generation across cutoffs."""
    tx, product = _make_synthetic_data()
    lines = merch_lines(tx, product)
    hdc = household_day_category(lines, set())

    # Cutoffs: 70 and 120
    # At cutoff 70, HH 1 only bought GROCERY (not PRODUCE). At cutoff 120, bought PRODUCE.
    flags = department_flags(hdc, cutoffs=[70, 120], departments=["GROCERY", "PRODUCE"])
    assert "bought_grocery_c" in flags.columns
    assert "bought_produce_c" in flags.columns

    hh1_c70 = flags[(flags["household_key"] == 1) & (flags["cutoff_day"] == 70)].iloc[0]
    assert hh1_c70["bought_grocery_c"] == 1
    assert hh1_c70["bought_produce_c"] == 0

    hh1_c120 = flags[(flags["household_key"] == 1) & (flags["cutoff_day"] == 120)].iloc[0]
    assert hh1_c120["bought_grocery_c"] == 1
    assert hh1_c120["bought_produce_c"] == 1


def test_quantile_bin_edge_cases() -> None:
    """Test quantile_bin interval naming, ties/zeros, and null category placement."""
    s = pd.Series([0.0, 0.0, 0.0, 0.1, 0.25, 0.5, 0.8, np.nan])
    cat = quantile_bin(s, q=4)

    assert "null" in cat.categories
    # null is placed last
    assert cat.categories[-1] == "null"
    assert cat[-1] == "null"

    # Verify duplicate drop handling
    s_zeros = pd.Series([0.0, 0.0, 0.0, 0.0, np.nan])
    cat_zeros = quantile_bin(s_zeros, q=5)
    assert len(cat_zeros.categories) == 2


def test_evaluate_department_flags_excess() -> None:
    """Test evaluation of department flags excess over expected_rf."""
    panel = pd.DataFrame(
        {
            "household_key": [1, 2, 3, 4],
            "cutoff_day": [140, 140, 140, 140],
            "target": [1, 0, 1, 0],
            "expected_rf": [0.5, 0.5, 0.5, 0.5],
            "bought_grocery_c": [1, 1, 0, 0],
        }
    )
    res = evaluate_department_flags_excess(panel, ["GROCERY"])
    assert len(res) == 1
    assert res.iloc[0]["department"] == "GROCERY"
    assert "excess_pp" in res.columns
    assert "verdict" in res.columns

"""Unit tests for promotion and campaign exploratory analysis."""

from __future__ import annotations

import pandas as pd
import pytest

from propensity.eda.promotion import (
    campaign_reach_by_rf,
    campaign_summary,
    campaign_type_summary,
    commodity_promo_summary,
    department_promo_summary,
    promo_dept_week,
    weekly_promo_summary,
)


def test_weekly_promo_summary_calculation_and_filtering() -> None:
    """Verify weekly promo metrics calculation and non-merchandise exclusion."""
    # Week 1: 2 merch lines (one promo spend 30, one regular spend 70) and 1 non-merch line
    lines = pd.DataFrame(
        [
            {
                "WEEK_NO": 1,
                "DAY": 5,
                "BASKET_ID": 101,
                "SALES_VALUE": 30.0,
                "RETAIL_DISC": -10.0,
                "COUPON_DISC": 0.0,
                "COUPON_MATCH_DISC": 0.0,
                "gross": 40.0,
                "is_merch": True,
                "is_promo": True,
                "has_coupon": False,
            },
            {
                "WEEK_NO": 1,
                "DAY": 6,
                "BASKET_ID": 102,
                "SALES_VALUE": 70.0,
                "RETAIL_DISC": 0.0,
                "COUPON_DISC": -5.0,
                "COUPON_MATCH_DISC": 0.0,
                "gross": 70.0,
                "is_merch": True,
                "is_promo": False,
                "has_coupon": True,
            },
            {
                # Non-merch line (should be completely excluded)
                "WEEK_NO": 1,
                "DAY": 5,
                "BASKET_ID": 103,
                "SALES_VALUE": 500.0,
                "RETAIL_DISC": -50.0,
                "COUPON_DISC": 0.0,
                "COUPON_MATCH_DISC": 0.0,
                "gross": 550.0,
                "is_merch": False,
                "is_promo": True,
                "has_coupon": False,
            },
        ]
    )

    res = weekly_promo_summary(lines)
    assert len(res) == 1
    row = res.iloc[0]
    assert row["sales"] == 100.0  # 30 + 70 (excluding 500)
    assert row["promo_spend"] == 30.0
    assert row["promo_spend_share"] == 0.30
    assert row["gross"] == 110.0
    assert pytest.approx(row["discount_depth"], rel=1e-3) == 10.0 / 110.0
    assert pytest.approx(row["coupon_disc_share"], rel=1e-3) == 5.0 / 110.0
    assert row["n_baskets"] == 2
    assert row["coupon_basket_share"] == 0.50


def test_department_and_commodity_promo_summary() -> None:
    """Verify department and commodity promo aggregation."""
    lines = pd.DataFrame(
        [
            {
                "DEPARTMENT": "GROCERY",
                "COMMODITY_DESC": "MILK",
                "SALES_VALUE": 50.0,
                "RETAIL_DISC": -10.0,
                "COUPON_DISC": 0.0,
                "COUPON_MATCH_DISC": 0.0,
                "gross": 60.0,
                "is_merch": True,
                "is_promo": True,
            },
            {
                "DEPARTMENT": "GROCERY",
                "COMMODITY_DESC": "BREAD",
                "SALES_VALUE": 50.0,
                "RETAIL_DISC": 0.0,
                "COUPON_DISC": 0.0,
                "COUPON_MATCH_DISC": 0.0,
                "gross": 50.0,
                "is_merch": True,
                "is_promo": False,
            },
        ]
    )

    dept = department_promo_summary(lines)
    assert len(dept) == 1
    assert dept.iloc[0]["promo_spend_share"] == 0.50

    comm = commodity_promo_summary(lines, min_spend_share=0.01)
    assert len(comm) == 2
    assert comm.iloc[0]["commodity"] == "MILK"
    assert comm.iloc[0]["promo_spend_share"] == 1.0


def test_campaign_summary_deduplication_and_redemption() -> None:
    """Verify campaign_summary deduplicates DESCRIPTION columns and computes correct metrics."""
    desc = pd.DataFrame(
        [
            {"CAMPAIGN": 1, "DESCRIPTION": "TypeA", "START_DAY": 10, "END_DAY": 40},
            {"CAMPAIGN": 2, "DESCRIPTION": "TypeB", "START_DAY": 50, "END_DAY": 70},
        ]
    )
    table = pd.DataFrame(
        [
            {"CAMPAIGN": 1, "DESCRIPTION": "TypeA", "household_key": 101},
            {"CAMPAIGN": 1, "DESCRIPTION": "TypeA", "household_key": 102},
            {"CAMPAIGN": 2, "DESCRIPTION": "TypeB", "household_key": 103},
        ]
    )
    redempt = pd.DataFrame(
        [
            {"CAMPAIGN": 1, "household_key": 101, "DAY": 25},
        ]
    )

    summary = campaign_summary(desc, table, redempt)
    assert len(summary) == 2
    # Check no duplicate columns like DESCRIPTION_x / DESCRIPTION_y
    assert "type" in summary.columns
    c1 = summary[summary["CAMPAIGN"] == 1].iloc[0]
    assert c1["n_hh_received"] == 2
    assert c1["n_hh_redeemed"] == 1
    assert c1["redemption_rate"] == 0.50
    assert c1["duration"] == 31  # 40 - 10 + 1

    t_sum = campaign_type_summary(summary)
    assert len(t_sum) == 2


def test_campaign_reach_by_rf() -> None:
    """Verify campaign reach across R and F strata."""
    panel = pd.DataFrame(
        {
            "F_bin": ["Q1", "Q1", "Q2", "Q2"],
            "recency_bin": ["1–7", "8–14", "1–7", "8–14"],
            "target": [1, 0, 1, 0],
            "in_active_campaign_c": [1, 0, 1, 1],
        }
    )
    reach = campaign_reach_by_rf(panel)
    assert "by_f" in reach
    assert "by_r" in reach
    assert reach["by_f"].loc[reach["by_f"]["F_bin"] == "Q1", "reach_rate"].iloc[0] == 0.50
    assert reach["by_f"].loc[reach["by_f"]["F_bin"] == "Q2", "reach_rate"].iloc[0] == 1.00

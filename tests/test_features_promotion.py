"""Unit tests for promotion and campaign feature engineering across cutoffs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propensity.features.promotion import campaign_features, promo_features


def test_promo_features_asof_cutoff_and_no_leakage() -> None:
    """Verify promo_features strictly uses DAY < c (no future leakage)."""
    # HH 1: buys on day 50 (merch spend 20, promo spend 20, retail_disc -5)
    # buys on day 100 (merch spend 30, regular, coupon used)
    # buys on day 150 (promo on exact day 150 - MUST NOT leak into cutoff 150)
    lines = pd.DataFrame(
        [
            {
                "household_key": 1,
                "DAY": 50,
                "BASKET_ID": 101,
                "SALES_VALUE": 20.0,
                "gross": 25.0,
                "RETAIL_DISC": -5.0,
                "COUPON_DISC": 0.0,
                "is_merch": True,
                "is_promo": True,
                "has_coupon": False,
            },
            {
                "household_key": 1,
                "DAY": 100,
                "BASKET_ID": 102,
                "SALES_VALUE": 30.0,
                "gross": 30.0,
                "RETAIL_DISC": 0.0,
                "COUPON_DISC": -2.0,
                "is_merch": True,
                "is_promo": False,
                "has_coupon": True,
            },
            {
                # Exact day 150 transaction
                "household_key": 1,
                "DAY": 150,
                "BASKET_ID": 103,
                "SALES_VALUE": 100.0,
                "gross": 120.0,
                "RETAIL_DISC": -20.0,
                "COUPON_DISC": 0.0,
                "is_merch": True,
                "is_promo": True,
                "has_coupon": False,
            },
        ]
    )

    # Cutoffs 70 and 150
    feats = promo_features(lines, cutoffs=[70, 150])
    assert len(feats) == 2

    # At cutoff 70: only day 50 is available
    c70 = feats[feats["cutoff_day"] == 70].iloc[0]
    assert c70["promo_spend_share_c"] == 1.0
    assert pytest.approx(c70["discount_depth_c"], rel=1e-3) == 5.0 / 25.0
    assert c70["used_coupon_c"] == 0
    assert c70["coupon_basket_share_c"] == 0.0

    # At cutoff 150: days 50 and 100 are available; day 150 MUST NOT be included!
    c150 = feats[feats["cutoff_day"] == 150].iloc[0]
    # Total spend = 20 + 30 = 50. Promo spend = 20. Gross = 25 + 30 = 55. Retail disc = 5.
    assert pytest.approx(c150["promo_spend_share_c"], rel=1e-3) == 20.0 / 50.0
    assert pytest.approx(c150["discount_depth_c"], rel=1e-3) == 5.0 / 55.0
    assert c150["used_coupon_c"] == 1
    assert pytest.approx(c150["coupon_basket_share_c"], rel=1e-3) == 1.0 / 2.0


def test_campaign_features_boundary_conditions() -> None:
    """Verify campaign active boundaries: START_DAY = c is NOT active; END_DAY = c IS active; multiple types -> Nhiều loại."""
    # Campaign 1: START_DAY=100, END_DAY=150 (Type A)
    # Campaign 2: START_DAY=140, END_DAY=200 (Type B)
    desc = pd.DataFrame(
        [
            {"CAMPAIGN": 1, "DESCRIPTION": "TypeA", "START_DAY": 100, "END_DAY": 150},
            {"CAMPAIGN": 2, "DESCRIPTION": "TypeB", "START_DAY": 140, "END_DAY": 200},
        ]
    )
    table = pd.DataFrame(
        [
            {"CAMPAIGN": 1, "household_key": 10},
            {"CAMPAIGN": 2, "household_key": 10},
        ]
    )
    redempt = pd.DataFrame(
        [
            {"CAMPAIGN": 1, "household_key": 10, "DAY": 140},
        ]
    )

    # Panel with cutoffs 100, 140, 150, 201
    panel = pd.DataFrame(
        {
            "household_key": [10, 10, 10, 10],
            "cutoff_day": [100, 140, 150, 201],
        }
    )

    cf = campaign_features(table, desc, redempt, panel)
    assert len(cf) == 4

    # 1. At cutoff 100: START_DAY = 100 -> NOT active yet (START_DAY < c is false)
    c100 = cf[cf["cutoff_day"] == 100].iloc[0]
    assert c100["in_active_campaign_c"] == 0
    assert str(c100["active_campaign_type_c"]) == "Không có"
    assert c100["n_campaigns_started_c"] == 0
    assert c100["redeemed_coupon_c"] == 0

    # 2. At cutoff 140:
    # Campaign 1 is active (100 < 140 <= 150).
    # Campaign 2: START_DAY = 140 -> NOT active yet!
    # Redemption on DAY 140: not < 140, so redeemed_coupon_c MUST be 0!
    c140 = cf[cf["cutoff_day"] == 140].iloc[0]
    assert c140["in_active_campaign_c"] == 1
    assert str(c140["active_campaign_type_c"]) == "Type A"
    assert c140["n_campaigns_started_c"] == 1
    assert c140["redeemed_coupon_c"] == 0  # Day 140 redemption not strictly before cutoff 140

    # 3. At cutoff 150:
    # Campaign 1: END_DAY = 150 -> Still active!
    # Campaign 2: START_DAY = 140 -> Active!
    # Both Type A and Type B active -> "Nhiều loại"!
    # Redemption on day 140 < 150 -> redeemed_coupon_c MUST be 1!
    c150 = cf[cf["cutoff_day"] == 150].iloc[0]
    assert c150["in_active_campaign_c"] == 1
    assert str(c150["active_campaign_type_c"]) == "Nhiều loại"
    assert c150["n_campaigns_started_c"] == 2
    assert c150["redeemed_coupon_c"] == 1

    # 4. At cutoff 201: both ended
    c201 = cf[cf["cutoff_day"] == 201].iloc[0]
    assert c201["in_active_campaign_c"] == 0
    assert str(c201["active_campaign_type_c"]) == "Không có"

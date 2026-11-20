"""Unit tests for basket-level feature extraction and aggregations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propensity.features.basket import (
    basket_features,
    basket_size_bands,
    basket_table,
    day_basket_table,
)


def test_basket_table_aggregation() -> None:
    """Verify basket_table correctly computes merch_spend, n_products, and has_fuel."""
    lines = pd.DataFrame(
        {
            "household_key": [1, 1, 1, 2],
            "BASKET_ID": [101, 101, 102, 201],
            "DAY": [10, 10, 15, 20],
            "PRODUCT_ID": [1001, 1002, 1001, 9999],
            "SALES_VALUE": [12.5, 7.5, 5.0, 50.0],
            "is_merch": [True, True, True, False],
            "is_fuel": [False, False, False, True],
        }
    )

    baskets = basket_table(lines)
    assert len(baskets) == 3

    b101 = baskets[baskets["BASKET_ID"] == 101].iloc[0]
    assert b101["household_key"] == 1
    assert b101["DAY"] == 10
    assert np.isclose(b101["merch_spend"], 20.0)
    assert b101["n_products"] == 2
    assert not b101["has_fuel"]

    b201 = baskets[baskets["BASKET_ID"] == 201].iloc[0]
    assert b201["household_key"] == 2
    assert np.isclose(b201["merch_spend"], 0.0)
    assert b201["n_products"] == 0
    assert b201["has_fuel"]


def test_day_basket_table() -> None:
    """Verify day_basket_table rolls multiple baskets into household x DAY."""
    baskets = pd.DataFrame(
        {
            "household_key": [1, 1, 2],
            "BASKET_ID": [101, 102, 201],
            "DAY": [10, 10, 20],
            "merch_spend": [20.0, 15.0, 40.0],
            "n_products": [3, 2, 5],
            "has_fuel": [False, False, True],
        }
    )

    day_tbl = day_basket_table(baskets)
    assert len(day_tbl) == 2

    d10 = day_tbl[(day_tbl["household_key"] == 1) & (day_tbl["DAY"] == 10)].iloc[0]
    assert np.isclose(d10["day_spend"], 35.0)
    assert d10["day_products"] == 5


def test_basket_size_bands() -> None:
    """Verify basket_size_bands groups merchandise baskets into 5 bands."""
    baskets = pd.DataFrame(
        {
            "household_key": [1, 1, 2, 2, 3],
            "BASKET_ID": [1, 2, 3, 4, 5],
            "DAY": [1, 2, 3, 4, 5],
            "merch_spend": [10.0, 25.0, 50.0, 100.0, 200.0],
            "n_products": [1, 4, 8, 15, 30],
            "has_fuel": [False, False, False, False, False],
        }
    )

    bands = basket_size_bands(baskets)
    assert len(bands) == 5
    assert list(bands["band"]) == ["1–2", "3–5", "6–10", "11–20", "21+"]
    assert np.isclose(bands["pct_baskets"].sum(), 1.0)
    assert np.isclose(bands["pct_spend"].sum(), 1.0)
    assert bands.loc[bands["band"] == "1–2", "n_baskets"].iloc[0] == 1
    assert bands.loc[bands["band"] == "21+", "n_baskets"].iloc[0] == 1


def test_basket_features_strict_point_in_time() -> None:
    """Verify basket_features respects as-of cutoff and returns null ratios when <3 purchase days."""
    # HH 1: 3 purchase days before c=50: days 10 (prods=2, spend=10), 20 (prods=4, spend=20), 30 (prods=6, spend=30)
    # Day 60 is AFTER cutoff 50 (must be excluded!)
    # HH 2: only 2 purchase days before c=50: days 15, 25 -> must have NaN last_basket_ratio_c
    day_tbl = pd.DataFrame(
        {
            "household_key": [1, 1, 1, 1, 2, 2],
            "DAY": [10, 20, 30, 60, 15, 25],
            "day_spend": [10.0, 20.0, 30.0, 100.0, 5.0, 15.0],
            "day_products": [2, 4, 6, 20, 1, 3],
        }
    )

    res = basket_features(day_tbl, cutoffs=[50])
    assert len(res) == 2

    hh1 = res[res["household_key"] == 1].iloc[0]
    # In DAY < 50: prods are [2, 4, 6] -> median is 4.0; last is 6.
    assert np.isclose(hh1["median_products_per_day_c"], 4.0)
    assert hh1["last_day_products_c"] == 6
    assert np.isclose(hh1["last_basket_ratio_c"], 6.0 / 4.0)

    hh2 = res[res["household_key"] == 2].iloc[0]
    # Only 2 days -> ratio must be NaN
    assert pd.isna(hh2["last_basket_ratio_c"])
    assert pd.isna(hh2["last_spend_ratio_c"])

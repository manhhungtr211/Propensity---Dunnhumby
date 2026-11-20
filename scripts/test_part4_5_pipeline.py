"""Probe and test calculation pipeline for Part 4 + 5 (Basket and Temporal)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from propensity.features.category import CategoryConfig, merch_lines
from propensity.features.basket import (
    basket_table,
    day_basket_table,
    basket_size_bands,
    basket_features,
)
from propensity.features.temporal import (
    temporal_features,
    trend_bin,
    lapse_bin,
    weekly_activity,
    cutoff_return_rates,
)
from propensity.eda.customer_label_relation import (
    CustomerLabelConfig,
    build_customer_cutoff_panel,
)
from propensity.eda.propensity_rates import (
    rf_baseline,
    excess_over_rf,
    add_verdicts,
    join_features,
    quantile_bin,
)

root = Path(__file__).resolve().parents[1]
raw_dir = root / "data" / "raw" / "Dunnhumby"

print("1. Loading raw data...")
tx = pd.read_csv(raw_dir / "transaction_data.csv")
prod = pd.read_csv(raw_dir / "product.csv")

cfg = CategoryConfig()
c_cfg = CustomerLabelConfig()

print("2. Building customer cutoff panel...")
panel = build_customer_cutoff_panel(tx, c_cfg)
rf_res = rf_baseline(panel)
panel = rf_res.panel

print("3. Merch lines & baskets...")
lines = merch_lines(tx, prod, cfg)
baskets = basket_table(lines)
bands = basket_size_bands(baskets)
print("\n=== Basket Size Bands ===")
print(bands)

cutoffs = list(range(140, 701, 7))

print("\n4. Basket features...")
day_tbl = day_basket_table(baskets, lines=lines)
bf = basket_features(day_tbl, cutoffs)
panel = join_features(panel, bf)

panel["basket_size_bin"] = quantile_bin(panel["median_products_per_day_c"], 5, prefix="Q")
q4b_excess = add_verdicts(excess_over_rf(panel, "basket_size_bin"))
print("\n=== Q4b Basket Size Excess ===")
print(q4b_excess[["basket_size_bin", "n_households", "observed", "expected", "excess_pp", "same_sign_share", "verdict"]])

panel["last_basket_bin"] = pd.cut(
    panel["last_basket_ratio_c"],
    bins=[-np.inf, 0.5, 1.0, 1.5, 2.5, np.inf],
    labels=["< 0.5 (Rất nhỏ)", "0.5–1.0 (Hơi nhỏ/bằng)", "1.0–1.5 (Hơi lớn)", "1.5–2.5 (Lớn)", "> 2.5 (Rất lớn)"],
)
q4c_excess = add_verdicts(excess_over_rf(panel, "last_basket_bin"))
print("\n=== Q4c Last Basket Ratio Excess ===")
print(q4c_excess[["last_basket_bin", "n_households", "observed", "expected", "excess_pp", "same_sign_share", "verdict"]])

# Compare with last_spend_ratio_c
panel["last_spend_bin"] = pd.cut(
    panel["last_spend_ratio_c"],
    bins=[-np.inf, 0.5, 1.0, 1.5, 2.5, np.inf],
    labels=["< 0.5 (Rất nhỏ)", "0.5–1.0 (Hơi nhỏ/bằng)", "1.0–1.5 (Hơi lớn)", "1.5–2.5 (Lớn)", "> 2.5 (Rất lớn)"],
)
q4c_spend_excess = add_verdicts(excess_over_rf(panel, "last_spend_bin"))
print("\n=== Q4c Last Spend Ratio Excess ===")
print(q4c_spend_excess[["last_spend_bin", "n_households", "observed", "expected", "excess_pp", "same_sign_share", "verdict"]])

print("\n5. Temporal activity & rates...")
weekly = weekly_activity(lines)
rates = cutoff_return_rates(panel)
print("Cutoff rates summary:")
print(f"Median rate: {rates['return_rate'].median():.4f}, unusual cutoffs: {rates['is_unusual'].sum()}")
if rates['is_unusual'].sum() > 0:
    print("Unusual cutoffs:", rates[rates['is_unusual']][["cutoff_day", "return_rate", "robust_z"]])

print("\n6. Temporal features (recent_ipt_ratio_c)...")
pdays = lines[["household_key", "DAY"]].drop_duplicates()
tf = temporal_features(pdays, cutoffs)
panel = join_features(panel, tf)

panel["ipt_trend_bin"] = trend_bin(panel["recent_ipt_ratio_c"])
q5b_excess = add_verdicts(excess_over_rf(panel, "ipt_trend_bin"))
print("\n=== Q5b IPT Trend Excess ===")
print(q5b_excess[["ipt_trend_bin", "n_households", "observed", "expected", "excess_pp", "same_sign_share", "verdict"]])

print("\n7. Q5c Lapse Ratio Excess...")
panel["lapse_cat_bin"] = lapse_bin(panel["lapse_ratio_c"])
q5c_excess = add_verdicts(excess_over_rf(panel, "lapse_cat_bin"))
print("\n=== Q5c Lapse Ratio Excess ===")
print(q5c_excess[["lapse_cat_bin", "n_households", "observed", "expected", "excess_pp", "same_sign_share", "verdict"]])

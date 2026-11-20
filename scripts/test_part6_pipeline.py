"""Probe and test calculation pipeline for Part 6 (Demographics)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from propensity.features.category import CategoryConfig, merch_lines
from propensity.eda.customer_label_relation import CustomerLabelConfig, build_customer_cutoff_panel
from propensity.eda.propensity_rates import rf_baseline, excess_over_rf, add_verdicts, FindingThresholds

root = Path(__file__).resolve().parents[1]
raw_dir = root / "data" / "raw" / "Dunnhumby"

print("1. Loading raw data...")
tx = pd.read_csv(raw_dir / "transaction_data.csv")
demo = pd.read_csv(raw_dir / "hh_demographic.csv")

DEMO_GROUPS = {
    "age": {"19-34": ["19-24", "25-34"], "35-44": ["35-44"], "45-54": ["45-54"], "55+": ["55-64", "65+"]},
    "income": {
        "<35K": ["Under 15K", "15-24K", "25-34K"],
        "35-49K": ["35-49K"],
        "50-74K": ["50-74K"],
        "75-124K": ["75-99K", "100-124K"],
        "125K+": ["125-149K", "150-174K", "175-199K", "200-249K", "250K+"],
    },
    "homeowner": {"Sở hữu": ["Homeowner", "Probable Owner"], "Thuê": ["Renter", "Probable Renter"], "Không rõ": ["Unknown"]},
    "marital": {"A": ["A"], "B": ["B"], "U": ["U"]},
    "hh_comp": {
        "Độc thân": ["Single Female", "Single Male"],
        "2 người lớn, không con": ["2 Adults No Kids"],
        "Có con": ["1 Adult Kids", "2 Adults Kids"],
        "Không rõ": ["Unknown"],
    },
    "hh_size": {"1": ["1"], "2": ["2"], "3": ["3"], "4+": ["4", "5+"]},
    "kids": {"Không/không rõ": ["None/Unknown"], "1": ["1"], "2+": ["2", "3+"]},
}

RAW_MAP = {
    "age": "AGE_DESC",
    "income": "INCOME_DESC",
    "homeowner": "HOMEOWNER_DESC",
    "marital": "MARITAL_STATUS_CODE",
    "hh_comp": "HH_COMP_DESC",
    "hh_size": "HOUSEHOLD_SIZE_DESC",
    "kids": "KID_CATEGORY_DESC",
}

def group_demographic(demo_df: pd.DataFrame, groups: dict) -> pd.DataFrame:
    df = demo_df[["household_key"]].copy()
    for attr, mapping in groups.items():
        raw_col = RAW_MAP[attr]
        inv_map = {}
        for target, raws in mapping.items():
            for r in raws:
                inv_map[r] = target
        # Check unmapped
        vals = demo_df[raw_col].dropna().unique()
        unmapped = set(vals) - set(inv_map.keys())
        if unmapped:
            raise ValueError(f"Giá trị không map được trong {raw_col}: {unmapped}")
        
        mapped = demo_df[raw_col].map(inv_map)
        df[f"demo_{attr}"] = pd.Categorical(mapped, categories=list(mapping.keys()), ordered=True)
    return df

demo_grouped = group_demographic(demo, DEMO_GROUPS)
print("demo_grouped shape:", demo_grouped.shape)
print("demo_grouped head:")
print(demo_grouped.head())

print("\n2. Building panel...")
c_cfg = CustomerLabelConfig()
panel = build_customer_cutoff_panel(tx, c_cfg)
rf_res = rf_baseline(panel)
panel = rf_res.panel

# Merge demographic
panel["has_demographic"] = panel["household_key"].isin(demo["household_key"]).astype(int)
for col in demo_grouped.columns:
    if col != "household_key":
        # merge
        hh_map = demo_grouped.set_index("household_key")[col].to_dict()
        panel[col] = panel["household_key"].map(hh_map)
        # Add category for missing
        cats = list(demo_grouped[col].cat.categories) + ["Không có dữ liệu"]
        panel[col] = pd.Categorical(panel[col].fillna("Không có dữ liệu"), categories=cats, ordered=True)

print("Panel shape after demo merge:", panel.shape)

print("\n3. Coverage Profile...")
cov_hh = panel.groupby("has_demographic")["household_key"].nunique()
cov_rows = panel.groupby("has_demographic").size()
cov_obs = panel.groupby("has_demographic")["target"].mean()
cov_exp = panel.groupby("has_demographic")["expected_rf"].mean()

print("Coverage summary:")
for k in [0, 1]:
    print(f"has_demo={k}: {cov_hh[k]} households, {cov_rows[k]} rows ({cov_rows[k]/len(panel):.2%}), obs={cov_obs[k]:.4f}, exp={cov_exp[k]:.4f}, excess={(cov_obs[k]-cov_exp[k])*100:.2f} pp")

f_col = "F_bin" if "F_bin" in panel.columns else "f_bin"
r_col = "recency_bin" if "recency_bin" in panel.columns else "r_bin"
print(f"\n{f_col} distribution by has_demographic:")
f_dist = pd.crosstab(panel["has_demographic"], panel[f_col], normalize="index") * 100
print(f_dist.round(1))

print(f"\n{r_col} distribution by has_demographic:")
r_dist = pd.crosstab(panel["has_demographic"], panel[r_col], normalize="index") * 100
print(r_dist.round(1))

panel["has_demo_str"] = panel["has_demographic"].map({0: "Không có demographic", 1: "Có demographic"}).astype("category")
q9a_excess = add_verdicts(excess_over_rf(panel, "has_demo_str"))
print("\n=== Q9a Excess over R x F ===")
print(q9a_excess[["has_demo_str", "n_households", "observed", "expected", "excess_pp", "same_sign_share", "verdict", "verdict_reason"]])

print("\n4. Attribute excess table on has_demographic == 1...")
panel_demo = panel[panel["has_demographic"] == 1].copy()
attrs = ["age", "income", "homeowner", "marital", "hh_comp", "hh_size", "kids"]
thr = FindingThresholds(min_row_share=0.02)

tables = []
for attr in attrs:
    col = f"demo_{attr}"
    # drop unused category 'Không có dữ liệu'
    panel_demo[col] = panel_demo[col].cat.remove_unused_categories()
    t = excess_over_rf(panel_demo, col)
    t["attribute"] = attr
    t["level"] = t[col].astype(str)
    tables.append(t)

all_attrs_df = pd.concat(tables, ignore_index=True)
all_attrs_df = add_verdicts(all_attrs_df, thr)
print("\n=== All Attributes Excess Table (Top rows & Verdicts) ===")
print(all_attrs_df[["attribute", "level", "n_households", "row_share", "observed", "expected", "excess_pp", "same_sign_share", "verdict"]].to_string())

print("\nVerdicts summary across all 28 levels:")
print(all_attrs_df["verdict"].value_counts())

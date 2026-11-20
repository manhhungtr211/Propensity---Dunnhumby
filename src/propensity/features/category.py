"""Category and commodity feature extraction and profiling across temporal cutoffs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
import pandas as pd

from propensity.eda.propensity_rates import add_verdicts, asof_aggregate, excess_over_rf
from propensity.purchases import valid_purchase_lines


@dataclass(frozen=True)
class CategoryConfig:
    """Configuration for category, commodity staple definitions, and department flags."""

    non_merch_departments: tuple[str, ...] = (
        "KIOSK-GAS",
        "MISC. TRANS.",
        "MISC SALES TRAN",
        "COUP/STR & MFG",
        "",
    )
    fuel_departments: tuple[str, ...] = ("KIOSK-GAS",)
    private_brand_value: str = "Private"
    staple_max_day: int = 139  # Dùng DAY ≤ 139 để định nghĩa thiết yếu và chọn department cho Q3e
    min_buyers: int = 30
    min_penetration: float = 0.50  # Gợi ý, chỉnh tối đa 1 lần sau khi xem scatter
    max_median_gap: float = 14.0
    dept_flag_penetration: tuple[float, float] = (0.20, 0.90)  # Khoảng penetration để chọn department cho Q3e


def merch_lines(
    tx: pd.DataFrame,
    product: pd.DataFrame,
    cfg: CategoryConfig | None = None,
) -> pd.DataFrame:
    """Enrich valid transaction lines with product metadata and merchandise/fuel/private flags.

    Lines with missing or unmatched PRODUCT_ID are marked as is_merch=False.
    """
    if cfg is None:
        cfg = CategoryConfig()

    valid = valid_purchase_lines(tx)

    req_prod_cols = {"PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC", "BRAND"}
    missing_prod = req_prod_cols - set(product.columns)
    if missing_prod:
        raise ValueError(f"Bảng product thiếu các cột: {missing_prod}")

    prod = product[["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC", "BRAND"]].copy()
    prod["DEPARTMENT"] = prod["DEPARTMENT"].fillna("").astype(str).str.strip()
    prod["COMMODITY_DESC"] = prod["COMMODITY_DESC"].fillna("UNKNOWN").astype(str).str.strip()
    prod["BRAND"] = prod["BRAND"].fillna("UNKNOWN").astype(str).str.strip()

    merged = valid.merge(prod, on="PRODUCT_ID", how="left")

    unmatched_mask = merged["DEPARTMENT"].isna()
    n_unmatched_ids = int(merged.loc[unmatched_mask, "PRODUCT_ID"].nunique())
    if n_unmatched_ids > 0:
        print(f"Số PRODUCT_ID không khớp trong product.csv: {n_unmatched_ids:,}")

    merged["DEPARTMENT"] = merged["DEPARTMENT"].fillna("").astype(str)
    merged["COMMODITY_DESC"] = merged["COMMODITY_DESC"].fillna("UNKNOWN").astype(str)
    merged["BRAND"] = merged["BRAND"].fillna("UNKNOWN").astype(str)

    merged["is_fuel"] = merged["DEPARTMENT"].isin(cfg.fuel_departments)
    merged["is_merch"] = (~merged["DEPARTMENT"].isin(cfg.non_merch_departments)) & (~unmatched_mask)
    merged["is_private"] = (merged["BRAND"] == cfg.private_brand_value) & merged["is_merch"]

    # Promotion and discount fields
    retail_disc = merged["RETAIL_DISC"] if "RETAIL_DISC" in merged.columns else 0.0
    coupon_disc = merged["COUPON_DISC"] if "COUPON_DISC" in merged.columns else 0.0
    coupon_match = merged["COUPON_MATCH_DISC"] if "COUPON_MATCH_DISC" in merged.columns else 0.0

    # In Dunnhumby, discounts are negative numbers, so subtracting adds back the discount value
    merged["gross"] = merged["SALES_VALUE"] - retail_disc - coupon_match
    merged["is_promo"] = retail_disc < 0
    merged["has_coupon"] = coupon_disc < 0

    return merged


def excluded_department_summary(lines: pd.DataFrame) -> pd.DataFrame:
    """Summarize non-merchandise departments excluded from core basket composition."""
    total_spend_all = float(lines["SALES_VALUE"].sum())
    total_hh_all = int(lines["household_key"].nunique())

    non_merch = lines[~lines["is_merch"]]
    records: list[dict[str, Any]] = []

    for dept, grp in non_merch.groupby("DEPARTMENT"):
        spend = float(grp["SALES_VALUE"].sum())
        spend_share = spend / total_spend_all if total_spend_all > 0 else 0.0
        n_hh = int(grp["household_key"].nunique())
        hh_share = n_hh / total_hh_all if total_hh_all > 0 else 0.0
        med_qty = float(grp["QUANTITY"].median()) if "QUANTITY" in grp.columns else np.nan

        top_comm = (
            grp.groupby("COMMODITY_DESC")["SALES_VALUE"]
            .sum()
            .nlargest(3)
            .index.tolist()
        )
        top_comm_str = ", ".join(top_comm) if top_comm else "N/A"

        records.append(
            {
                "department": dept if dept != "" else "(EMPTY/UNMATCHED)",
                "spend": spend,
                "spend_share": spend_share,
                "n_households": n_hh,
                "household_share": hh_share,
                "median_quantity": med_qty,
                "top_3_commodities": top_comm_str,
            }
        )

    res = pd.DataFrame(records)
    if not res.empty:
        res = res.sort_values("spend_share", ascending=False).reset_index(drop=True)
    return res


def department_summary(
    lines: pd.DataFrame,
    cfg: CategoryConfig | None = None,
) -> pd.DataFrame:
    """Summarize merchandise departments by spend share, Pareto accumulation, and penetration."""
    if cfg is None:
        cfg = CategoryConfig()

    merch = lines[lines["is_merch"]].copy()
    total_merch_spend = float(merch["SALES_VALUE"].sum())
    total_merch_hh = int(merch["household_key"].nunique())

    early = merch[merch["DAY"] <= cfg.staple_max_day]
    total_early_hh = int(early["household_key"].nunique()) if not early.empty else 0

    records: list[dict[str, Any]] = []
    for dept, grp in merch.groupby("DEPARTMENT"):
        spend = float(grp["SALES_VALUE"].sum())
        spend_share = spend / total_merch_spend if total_merch_spend > 0 else 0.0
        n_buyers = int(grp["household_key"].nunique())
        penetration = n_buyers / total_merch_hh if total_merch_hh > 0 else 0.0

        early_grp = early[early["DEPARTMENT"] == dept]
        n_buyers_early = int(early_grp["household_key"].nunique()) if not early_grp.empty else 0
        penetration_early = n_buyers_early / total_early_hh if total_early_hh > 0 else 0.0

        records.append(
            {
                "department": dept,
                "spend": spend,
                "spend_share": spend_share,
                "penetration": penetration,
                "penetration_early": penetration_early,
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("spend_share", ascending=False).reset_index(drop=True)
        df["cum_spend_share"] = df["spend_share"].cumsum()
    else:
        df["cum_spend_share"] = pd.Series(dtype="float64")

    return df


def commodity_stats(
    lines: pd.DataFrame,
    cfg: CategoryConfig | None = None,
) -> pd.DataFrame:
    """Calculate commodity level penetration, repeat purchase velocity, and median gap on DAY <= staple_max_day."""
    if cfg is None:
        cfg = CategoryConfig()

    sub = lines[lines["is_merch"] & (lines["DAY"] <= cfg.staple_max_day)].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "commodity",
                "department",
                "n_buyers",
                "penetration",
                "repeat_buyer_share",
                "median_gap",
                "spend",
                "spend_share",
            ]
        )

    total_hh_win = int(sub["household_key"].nunique())
    total_spend_win = float(sub["SALES_VALUE"].sum())

    # Pre-index household x commodity x DAY
    pdays_comm = (
        sub.groupby(["household_key", "COMMODITY_DESC", "DAY"], as_index=False)
        .agg(spend=("SALES_VALUE", "sum"), department=("DEPARTMENT", "first"))
    )

    records: list[dict[str, Any]] = []
    for comm, grp in pdays_comm.groupby("COMMODITY_DESC"):
        n_buyers = int(grp["household_key"].nunique())
        penetration = n_buyers / total_hh_win if total_hh_win > 0 else 0.0
        spend = float(grp["spend"].sum())
        spend_share = spend / total_spend_win if total_spend_win > 0 else 0.0
        dept_mode = grp["department"].mode().iloc[0] if not grp["department"].empty else ""

        # Calculate repeat buyer statistics
        buyer_pday_counts = grp.groupby("household_key")["DAY"].count()
        repeat_buyers = buyer_pday_counts[buyer_pday_counts >= 2].index
        n_repeat = len(repeat_buyers)
        repeat_buyer_share = n_repeat / n_buyers if n_buyers > 0 else 0.0

        if n_repeat > 0:
            repeat_grp = grp[grp["household_key"].isin(repeat_buyers)]
            buyer_median_gaps = [
                float(np.median(np.diff(np.sort(h_days))))
                for _, h_days in repeat_grp.groupby("household_key")["DAY"]
            ]
            median_gap = float(np.median(buyer_median_gaps))
        else:
            median_gap = np.nan

        records.append(
            {
                "commodity": comm,
                "department": dept_mode,
                "n_buyers": n_buyers,
                "penetration": penetration,
                "repeat_buyer_share": repeat_buyer_share,
                "median_gap": median_gap,
                "spend": spend,
                "spend_share": spend_share,
            }
        )

    res = pd.DataFrame(records)
    if not res.empty:
        res = res.sort_values("spend_share", ascending=False).reset_index(drop=True)
    return res


def define_staples(
    stats: pd.DataFrame,
    cfg: CategoryConfig | None = None,
) -> pd.DataFrame:
    """Classify commodities into staple vs non-staple based on buyers, penetration, and median gap."""
    if cfg is None:
        cfg = CategoryConfig()

    df = stats.copy()
    is_staple_list: list[bool] = []
    reasons: list[str] = []

    for _, row in df.iterrows():
        failed: list[str] = []
        if row["n_buyers"] < cfg.min_buyers:
            failed.append(f"buyers < {cfg.min_buyers}")
        if row["penetration"] < cfg.min_penetration:
            failed.append(f"penetration < {cfg.min_penetration:.0%}")
        if pd.isna(row["median_gap"]) or row["median_gap"] > cfg.max_median_gap:
            gap_val = f"{row['median_gap']:.1f}d" if pd.notna(row["median_gap"]) else "null"
            failed.append(f"gap {gap_val} > {cfg.max_median_gap:.0f}d")

        if not failed:
            is_staple_list.append(True)
            reasons.append("Staple")
        else:
            is_staple_list.append(False)
            reasons.append("; ".join(failed))

    df["is_staple"] = is_staple_list
    df["staple_reason"] = reasons
    return df


def select_flag_departments(
    dept_summary: pd.DataFrame,
    cfg: CategoryConfig | None = None,
) -> list[str]:
    """Select candidate departments for Q3e indicator flags based on early penetration range."""
    if cfg is None:
        cfg = CategoryConfig()

    min_p, max_p = cfg.dept_flag_penetration
    mask = (dept_summary["penetration_early"] >= min_p) & (dept_summary["penetration_early"] <= max_p)
    selected = dept_summary.loc[mask, "department"].tolist()
    return selected


def household_day_category(
    lines: pd.DataFrame,
    staple_set: set[str],
) -> pd.DataFrame:
    """Pre-aggregate transaction lines to (household_key, DAY, DEPARTMENT) for fast cutoff evaluations."""
    df = lines.copy()

    df["is_staple_line"] = df["is_merch"] & df["COMMODITY_DESC"].isin(staple_set)
    df["staple_spend"] = np.where(df["is_staple_line"], df["SALES_VALUE"], 0.0)
    df["private_spend"] = np.where(df["is_private"], df["SALES_VALUE"], 0.0)
    df["merch_spend"] = np.where(df["is_merch"], df["SALES_VALUE"], 0.0)
    df["fuel_spend"] = np.where(df["is_fuel"], df["SALES_VALUE"], 0.0)

    hdc = (
        df.groupby(["household_key", "DAY", "DEPARTMENT"], as_index=False)
        .agg(
            spend=("SALES_VALUE", "sum"),
            staple_spend=("staple_spend", "sum"),
            private_spend=("private_spend", "sum"),
            merch_spend=("merch_spend", "sum"),
            fuel_spend=("fuel_spend", "sum"),
            is_merch=("is_merch", "any"),
            is_fuel=("is_fuel", "any"),
        )
    )
    return hdc


def _compute_category_features_single(sub: pd.DataFrame) -> pd.DataFrame:
    """Internal feature calculation for a single point-in-time slice."""
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "household_key",
                "staple_spend_share_c",
                "dept_entropy_c",
                "n_departments_c",
                "private_label_share_c",
                "fuel_share_c",
                "buys_fuel_c",
            ]
        )

    # 1. Total spend sums per household
    totals = sub.groupby("household_key").agg(
        total_spend=("spend", "sum"),
        total_merch=("merch_spend", "sum"),
        total_staple=("staple_spend", "sum"),
        total_private=("private_spend", "sum"),
        total_fuel=("fuel_spend", "sum"),
    )

    # 2. Merchandise department diversity (entropy & count)
    sub_merch = sub[sub["is_merch"] & (sub["merch_spend"] > 0)]
    dept_spends = sub_merch.groupby(["household_key", "DEPARTMENT"])["merch_spend"].sum().reset_index()

    # Compute entropy per household
    entropies: dict[int, float] = {}
    dept_counts: dict[int, int] = {}

    for h, grp in dept_spends.groupby("household_key"):
        s_arr = grp["merch_spend"].to_numpy()
        dept_counts[h] = len(s_arr)
        tot = s_arr.sum()
        if tot > 0:
            p = s_arr / tot
            # Shannon entropy in bits (base 2)
            p_pos = p[p > 0]
            entropies[h] = float(-np.sum(p_pos * np.log2(p_pos)))
        else:
            entropies[h] = 0.0

    records: list[dict[str, Any]] = []
    for h, row in totals.iterrows():
        tot_spend = row["total_spend"]
        tot_merch = row["total_merch"]
        tot_staple = row["total_staple"]
        tot_private = row["total_private"]
        tot_fuel = row["total_fuel"]

        staple_share = tot_staple / tot_merch if tot_merch > 0 else np.nan
        private_share = tot_private / tot_merch if tot_merch > 0 else np.nan
        entropy = entropies.get(h, np.nan if tot_merch <= 0 else 0.0)
        n_dept = dept_counts.get(h, 0)
        fuel_share = tot_fuel / tot_spend if tot_spend > 0 else 0.0
        buys_fuel = 1 if fuel_share > 0 else 0

        records.append(
            {
                "household_key": h,
                "staple_spend_share_c": staple_share,
                "dept_entropy_c": entropy,
                "n_departments_c": n_dept,
                "private_label_share_c": private_share,
                "fuel_share_c": fuel_share,
                "buys_fuel_c": buys_fuel,
            }
        )

    return pd.DataFrame(records)


def category_features(
    hdc: pd.DataFrame,
    cutoffs: list[int] | tuple[int, ...] | np.ndarray | range | None = None,
) -> pd.DataFrame:
    """Calculate point-in-time category composition and diversity features across cutoffs (or for single slice)."""
    if cutoffs is None:
        return _compute_category_features_single(hdc)
    return asof_aggregate(hdc, cutoffs, _compute_category_features_single, day_col="DAY", key="household_key")


def _compute_department_flags_single(
    sub: pd.DataFrame,
    departments: list[str],
    clean_dept_map: dict[str, str],
) -> pd.DataFrame:
    """Internal department flag calculation for a single slice."""
    target_cols = list(clean_dept_map.values())
    if sub.empty:
        return pd.DataFrame(columns=["household_key"] + target_cols)

    sub_depts = sub[sub["DEPARTMENT"].isin(departments) & (sub["spend"] > 0)]
    hh_depts = sub_depts[["household_key", "DEPARTMENT"]].drop_duplicates()
    all_hh = sub["household_key"].unique()
    res_df = pd.DataFrame({"household_key": all_hh})

    for d, col_name in clean_dept_map.items():
        buyers_d = set(hh_depts.loc[hh_depts["DEPARTMENT"] == d, "household_key"])
        res_df[col_name] = res_df["household_key"].apply(lambda h: 1 if h in buyers_d else 0)

    return res_df


def department_flags(
    hdc: pd.DataFrame,
    cutoffs: list[int] | tuple[int, ...] | np.ndarray | range | None = None,
    departments: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate binary indicators (bought_<dept>_c) for selected departments as of each cutoff."""
    if departments is None:
        departments = []

    clean_dept_map = {
        d: f"bought_{re.sub(r'[^a-z0-9]+', '_', d.lower()).strip('_')}_c"
        for d in departments
    }

    if cutoffs is None:
        return _compute_department_flags_single(hdc, departments, clean_dept_map)

    def _agg(sub: pd.DataFrame) -> pd.DataFrame:
        return _compute_department_flags_single(sub, departments, clean_dept_map)

    return asof_aggregate(hdc, cutoffs, _agg, day_col="DAY", key="household_key")


def evaluate_department_flags_excess(
    panel: pd.DataFrame,
    flag_departments: list[str],
) -> pd.DataFrame:
    """Evaluate return rate and excess over R x F for buyers vs non-buyers of candidate departments."""
    dept_rows: list[dict[str, Any]] = []
    for d in flag_departments:
        col = f"bought_{re.sub(r'[^a-z0-9]+', '_', d.lower()).strip('_')}_c"
        if col not in panel.columns:
            continue
        col_bin = f"{col}_bin"
        temp_s = panel[col].map({0: "Chưa mua", 1: "Đã mua"}).astype("category")
        panel[col_bin] = temp_s
        ex = add_verdicts(excess_over_rf(panel, col_bin))
        row1 = ex[ex[col_bin] == "Đã mua"].iloc[0]
        dept_rows.append(
            {
                "department": d,
                "flag_col": col,
                "n_rows": int(row1["n_rows"]),
                "n_households": int(row1["n_households"]),
                "buyer_share": float(row1["row_share"]),
                "observed": float(row1["observed"]),
                "expected": float(row1["expected"]),
                "excess_pp": float(row1["excess_pp"]),
                "same_sign_share": float(row1["same_sign_share"]),
                "verdict": str(row1["verdict"]),
            }
        )

    res_df = pd.DataFrame(dept_rows)
    if not res_df.empty:
        res_df = res_df.sort_values("excess_pp", ascending=False).reset_index(drop=True)
    return res_df

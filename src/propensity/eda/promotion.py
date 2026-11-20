"""Descriptive analysis and aggregations for promotions and marketing campaigns."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def weekly_promo_summary(lines: pd.DataFrame) -> pd.DataFrame:
    """Calculate weekly merchandise sales, promotion spend share, discount depth, and coupon usage."""
    sub = lines[lines["is_merch"]].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "WEEK_NO",
                "sales",
                "gross",
                "promo_spend",
                "promo_spend_share",
                "discount_depth",
                "coupon_disc_share",
                "n_baskets",
                "coupon_basket_share",
            ]
        )

    if "WEEK_NO" not in sub.columns:
        sub["WEEK_NO"] = (sub["DAY"] - 1) // 7 + 1

    # Basket level coupon indicator
    basket_df = sub[["WEEK_NO", "BASKET_ID", "has_coupon"]].drop_duplicates()
    basket_weekly = basket_df.groupby("WEEK_NO").agg(
        n_baskets=("BASKET_ID", "nunique"),
        coupon_baskets=("has_coupon", "sum"),
    ).reset_index()

    grouped = sub.groupby("WEEK_NO").agg(
        sales=("SALES_VALUE", "sum"),
        gross=("gross", "sum"),
        promo_spend=("SALES_VALUE", lambda s: s[sub.loc[s.index, "is_promo"]].sum()),
        retail_disc=("RETAIL_DISC", lambda d: (-d).sum()),
        coupon_disc=("COUPON_DISC", lambda d: (-d).sum()),
    ).reset_index()

    res = grouped.merge(basket_weekly, on="WEEK_NO", how="left")
    res["promo_spend_share"] = np.where(res["sales"] > 0, res["promo_spend"] / res["sales"], 0.0)
    res["discount_depth"] = np.where(res["gross"] > 0, res["retail_disc"] / res["gross"], 0.0)
    res["coupon_disc_share"] = np.where(res["gross"] > 0, res["coupon_disc"] / res["gross"], 0.0)
    res["coupon_basket_share"] = np.where(res["n_baskets"] > 0, res["coupon_baskets"] / res["n_baskets"], 0.0)

    cols = [
        "WEEK_NO",
        "sales",
        "gross",
        "promo_spend",
        "promo_spend_share",
        "discount_depth",
        "coupon_disc_share",
        "n_baskets",
        "coupon_basket_share",
    ]
    return res[cols].sort_values("WEEK_NO").reset_index(drop=True)


def department_promo_summary(lines: pd.DataFrame) -> pd.DataFrame:
    """Summarize promotion spend share, discount depth, and coupon usage by department."""
    sub = lines[lines["is_merch"]].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "department",
                "spend",
                "spend_share",
                "promo_spend_share",
                "discount_depth",
                "coupon_disc_share",
            ]
        )

    total_merch_spend = float(sub["SALES_VALUE"].sum())

    records: list[dict[str, Any]] = []
    for dept, grp in sub.groupby("DEPARTMENT"):
        spend = float(grp["SALES_VALUE"].sum())
        spend_share = spend / total_merch_spend if total_merch_spend > 0 else 0.0

        promo_spend = float(grp.loc[grp["is_promo"], "SALES_VALUE"].sum())
        promo_share = promo_spend / spend if spend > 0 else 0.0

        gross = float(grp["gross"].sum())
        retail_disc = float((-grp["RETAIL_DISC"]).sum())
        coupon_disc = float((-grp["COUPON_DISC"]).sum())

        depth = retail_disc / gross if gross > 0 else 0.0
        coupon_share = coupon_disc / gross if gross > 0 else 0.0

        records.append(
            {
                "department": dept,
                "spend": spend,
                "spend_share": spend_share,
                "promo_spend_share": promo_share,
                "discount_depth": depth,
                "coupon_disc_share": coupon_share,
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("promo_spend_share", ascending=False).reset_index(drop=True)
    return df


def commodity_promo_summary(
    lines: pd.DataFrame,
    min_spend_share: float = 0.005,
) -> pd.DataFrame:
    """Summarize promotions for commodities with at least min_spend_share of total merchandise spend."""
    sub = lines[lines["is_merch"]].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "commodity",
                "department",
                "spend",
                "spend_share",
                "promo_spend_share",
                "discount_depth",
                "coupon_disc_share",
            ]
        )

    total_merch_spend = float(sub["SALES_VALUE"].sum())

    records: list[dict[str, Any]] = []
    for comm, grp in sub.groupby("COMMODITY_DESC"):
        spend = float(grp["SALES_VALUE"].sum())
        spend_share = spend / total_merch_spend if total_merch_spend > 0 else 0.0

        if spend_share < min_spend_share:
            continue

        dept = grp["DEPARTMENT"].mode().iloc[0] if not grp["DEPARTMENT"].empty else "UNKNOWN"
        promo_spend = float(grp.loc[grp["is_promo"], "SALES_VALUE"].sum())
        promo_share = promo_spend / spend if spend > 0 else 0.0

        gross = float(grp["gross"].sum())
        retail_disc = float((-grp["RETAIL_DISC"]).sum())
        coupon_disc = float((-grp["COUPON_DISC"]).sum())

        depth = retail_disc / gross if gross > 0 else 0.0
        coupon_share = coupon_disc / gross if gross > 0 else 0.0

        records.append(
            {
                "commodity": comm,
                "department": dept,
                "spend": spend,
                "spend_share": spend_share,
                "promo_spend_share": promo_share,
                "discount_depth": depth,
                "coupon_disc_share": coupon_share,
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("promo_spend_share", ascending=False).reset_index(drop=True)
    return df


def promo_dept_week(lines: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Create a department x week matrix of promo_spend_share for top_n departments by merchandise spend."""
    sub = lines[lines["is_merch"]].copy()
    if sub.empty:
        return pd.DataFrame()

    if "WEEK_NO" not in sub.columns:
        sub["WEEK_NO"] = (sub["DAY"] - 1) // 7 + 1

    # Identify top_n departments by total spend
    dept_spends = sub.groupby("DEPARTMENT")["SALES_VALUE"].sum().nlargest(top_n).index.tolist()
    sub_top = sub[sub["DEPARTMENT"].isin(dept_spends)]

    # Compute promo_spend and total spend per department x week
    agg = sub_top.groupby(["DEPARTMENT", "WEEK_NO"]).agg(
        spend=("SALES_VALUE", "sum"),
        promo_spend=("SALES_VALUE", lambda s: s[sub_top.loc[s.index, "is_promo"]].sum()),
    ).reset_index()

    agg["promo_spend_share"] = np.where(agg["spend"] > 0, agg["promo_spend"] / agg["spend"], 0.0)

    matrix = pd.pivot_table(
        agg,
        index="DEPARTMENT",
        columns="WEEK_NO",
        values="promo_spend_share",
        aggfunc="first",
    ).reindex(index=dept_spends).fillna(0.0)

    return matrix


def campaign_summary(
    campaign_desc: pd.DataFrame,
    campaign_table: pd.DataFrame,
    coupon_redempt: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate summary metrics for each marketing campaign (Type, Start, End, Duration, Reach, Redemption)."""
    # Group campaign assignments per campaign
    ct_grouped = campaign_table.groupby("CAMPAIGN")["household_key"].nunique().reset_index(name="n_hh_received")

    # Group coupon redemptions per campaign
    cr_grouped = coupon_redempt.groupby("CAMPAIGN")["household_key"].nunique().reset_index(name="n_hh_redeemed")

    # Merge ONLY on CAMPAIGN to avoid duplicate DESCRIPTION columns
    merged = campaign_desc.merge(ct_grouped, on="CAMPAIGN", how="left").merge(cr_grouped, on="CAMPAIGN", how="left")

    merged["n_hh_received"] = merged["n_hh_received"].fillna(0).astype(int)
    merged["n_hh_redeemed"] = merged["n_hh_redeemed"].fillna(0).astype(int)
    merged["redemption_rate"] = np.where(
        merged["n_hh_received"] > 0,
        merged["n_hh_redeemed"] / merged["n_hh_received"],
        0.0,
    )
    merged["duration"] = merged["END_DAY"] - merged["START_DAY"] + 1

    res = merged.rename(
        columns={
            "DESCRIPTION": "type",
            "START_DAY": "start",
            "END_DAY": "end",
        }
    )[["CAMPAIGN", "type", "start", "end", "duration", "n_hh_received", "n_hh_redeemed", "redemption_rate"]]

    return res.sort_values(["start", "CAMPAIGN"]).reset_index(drop=True)


def campaign_type_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize marketing campaigns grouped by campaign type (TypeA, TypeB, TypeC)."""
    if summary.empty:
        return pd.DataFrame(
            columns=["type", "n_campaigns", "median_duration", "total_hh_received", "total_hh_redeemed", "redemption_rate"]
        )

    grouped = summary.groupby("type").agg(
        n_campaigns=("CAMPAIGN", "count"),
        median_duration=("duration", "median"),
        total_hh_received=("n_hh_received", "sum"),
        total_hh_redeemed=("n_hh_redeemed", "sum"),
    ).reset_index()

    grouped["redemption_rate"] = np.where(
        grouped["total_hh_received"] > 0,
        grouped["total_hh_redeemed"] / grouped["total_hh_received"],
        0.0,
    )
    return grouped.sort_values("type").reset_index(drop=True)


def campaign_reach_by_rf(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Calculate the share of observations in an active campaign broken down by F_bin and recency_bin (evidence of selection bias)."""
    if "in_active_campaign_c" not in panel.columns:
        raise ValueError("Panel thiếu cột 'in_active_campaign_c'")

    # By Frequency bin
    f_col = "F_bin" if "F_bin" in panel.columns else "f_bin"
    r_col = "recency_bin" if "recency_bin" in panel.columns else "r_bin"

    f_reach = (
        panel.groupby(f_col, observed=False)
        .agg(
            n_rows=("target", "count"),
            active_rows=("in_active_campaign_c", "sum"),
            reach_rate=("in_active_campaign_c", "mean"),
        )
        .reset_index()
    )

    r_reach = (
        panel.groupby(r_col, observed=False)
        .agg(
            n_rows=("target", "count"),
            active_rows=("in_active_campaign_c", "sum"),
            reach_rate=("in_active_campaign_c", "mean"),
        )
        .reset_index()
    )

    return {"by_f": f_reach, "by_r": r_reach}

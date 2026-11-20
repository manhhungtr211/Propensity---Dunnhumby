"""Point-in-time promotion, coupon, and marketing campaign feature engineering across cutoffs."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from propensity.eda.propensity_rates import asof_aggregate


def _compute_promo_features_single(sub_hday: pd.DataFrame) -> pd.DataFrame:
    """Internal calculation of promotion and coupon features for a single point-in-time slice."""
    if sub_hday.empty:
        return pd.DataFrame(
            columns=[
                "household_key",
                "promo_spend_share_c",
                "discount_depth_c",
                "coupon_basket_share_c",
                "used_coupon_c",
            ]
        )

    grouped = sub_hday.groupby("household_key").agg(
        merch_spend=("merch_spend", "sum"),
        promo_spend=("promo_spend", "sum"),
        gross=("gross", "sum"),
        retail_disc=("retail_disc", "sum"),
        baskets=("baskets", "sum"),
        coupon_baskets=("coupon_baskets", "sum"),
    )

    records: list[dict[str, Any]] = []
    for h, row in grouped.iterrows():
        m_spend = row["merch_spend"]
        p_spend = row["promo_spend"]
        gross = row["gross"]
        r_disc = row["retail_disc"]
        n_bask = row["baskets"]
        c_bask = row["coupon_baskets"]

        promo_share = p_spend / m_spend if m_spend > 0 else np.nan
        disc_depth = r_disc / gross if gross > 0 else np.nan
        coupon_bask_share = c_bask / n_bask if n_bask > 0 else 0.0
        used_coupon = 1 if c_bask > 0 else 0

        records.append(
            {
                "household_key": h,
                "promo_spend_share_c": promo_share,
                "discount_depth_c": disc_depth,
                "coupon_basket_share_c": coupon_bask_share,
                "used_coupon_c": used_coupon,
            }
        )

    return pd.DataFrame(records)


def promo_features(
    lines: pd.DataFrame,
    cutoffs: list[int] | tuple[int, ...] | range | None = None,
) -> pd.DataFrame:
    """Calculate point-in-time promotion features (promo_spend_share_c, discount_depth_c, coupon usage) strictly as-of DAY < c."""
    # Filter to merchandise lines
    sub = lines[lines["is_merch"]].copy()
    if sub.empty:
        cols = ["household_key", "promo_spend_share_c", "discount_depth_c", "coupon_basket_share_c", "used_coupon_c"]
        if cutoffs is not None:
            cols.append("cutoff_day")
        return pd.DataFrame(columns=cols)

    # Pre-aggregate to household x DAY to ensure maximum execution speed
    # Basket coupon indicator per line
    has_c = sub["has_coupon"] if "has_coupon" in sub.columns else (sub["COUPON_DISC"] < 0)
    is_p = sub["is_promo"] if "is_promo" in sub.columns else (sub["RETAIL_DISC"] < 0)
    ret_d = (-sub["RETAIL_DISC"]) if "RETAIL_DISC" in sub.columns else 0.0

    sub_copy = sub[["household_key", "DAY", "BASKET_ID", "SALES_VALUE", "gross"]].copy()
    sub_copy["has_coupon"] = has_c
    sub_copy["is_promo"] = is_p
    sub_copy["retail_disc"] = ret_d
    sub_copy["promo_spend"] = np.where(sub_copy["is_promo"], sub_copy["SALES_VALUE"], 0.0)

    # Basket level aggregation per household x DAY
    basket_df = sub_copy.groupby(["household_key", "DAY", "BASKET_ID"])["has_coupon"].any().reset_index()
    basket_day = basket_df.groupby(["household_key", "DAY"]).agg(
        baskets=("BASKET_ID", "count"),
        coupon_baskets=("has_coupon", "sum"),
    ).reset_index()

    spend_day = sub_copy.groupby(["household_key", "DAY"]).agg(
        merch_spend=("SALES_VALUE", "sum"),
        promo_spend=("promo_spend", "sum"),
        gross=("gross", "sum"),
        retail_disc=("retail_disc", "sum"),
    ).reset_index()

    hday = spend_day.merge(basket_day, on=["household_key", "DAY"], how="left")

    if cutoffs is None:
        return _compute_promo_features_single(hday)

    return asof_aggregate(hday, cutoffs, _compute_promo_features_single, day_col="DAY", key="household_key")


def campaign_features(
    campaign_table: pd.DataFrame,
    campaign_desc: pd.DataFrame,
    coupon_redempt: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate point-in-time marketing campaign indicators strictly as-of DAY < c.

    Parameters
    ----------
    campaign_table : DataFrame
        household_key, CAMPAIGN
    campaign_desc : DataFrame
        CAMPAIGN, DESCRIPTION, START_DAY, END_DAY
    coupon_redempt : DataFrame
        household_key, DAY, CAMPAIGN
    panel : DataFrame
        household_key, cutoff_day

    Returns
    -------
    DataFrame with columns: household_key, cutoff_day, in_active_campaign_c, active_campaign_type_c, n_campaigns_started_c, redeemed_coupon_c
    """
    req_panel_cols = {"household_key", "cutoff_day"}
    missing = req_panel_cols - set(panel.columns)
    if missing:
        raise ValueError(f"Panel thiếu các cột bắt buộc: {missing}")

    # Standardize campaign description names (TypeA -> Type A)
    cd = campaign_desc[["CAMPAIGN", "DESCRIPTION", "START_DAY", "END_DAY"]].copy()
    cd["type_clean"] = cd["DESCRIPTION"].astype(str).str.strip().apply(
        lambda t: f"Type {t[-1]}" if t.startswith("Type") and len(t) > 4 else t
    )

    # Household-campaign assignments
    assignments = campaign_table[["household_key", "CAMPAIGN"]].drop_duplicates().merge(
        cd[["CAMPAIGN", "type_clean", "START_DAY", "END_DAY"]],
        on="CAMPAIGN",
        how="inner",
    )

    # 1. Minimum redemption day per household from coupon_redempt
    min_redempt = coupon_redempt.groupby("household_key")["DAY"].min().to_dict()

    # Process by cutoff to build features
    all_cutoffs = sorted(panel["cutoff_day"].unique())
    panel_records: list[dict[str, Any]] = []

    # Map household keys present in panel for each cutoff
    hh_by_cutoff = panel.groupby("cutoff_day")["household_key"].unique().to_dict()

    for c in all_cutoffs:
        hhs = hh_by_cutoff.get(c, np.array([]))
        if len(hhs) == 0:
            continue

        # Active campaigns at cutoff c: START_DAY < c <= END_DAY
        active_c = assignments[(assignments["START_DAY"] < c) & (c <= assignments["END_DAY"])]
        # Started campaigns before cutoff c: START_DAY < c
        started_c = assignments[assignments["START_DAY"] < c]

        # Group active types per household
        active_types_map: dict[int, list[str]] = (
            active_c.groupby("household_key")["type_clean"].unique().to_dict()
        )

        # Count started campaigns per household
        started_counts_map: dict[int, int] = started_c.groupby("household_key").size().to_dict()

        for h in hhs:
            # Active campaign check
            act_types = active_types_map.get(h, None)
            if act_types is None or len(act_types) == 0:
                in_active = 0
                camp_type = "Không có"
            else:
                in_active = 1
                if len(act_types) == 1:
                    camp_type = act_types[0]
                else:
                    camp_type = "Nhiều loại"

            n_started = started_counts_map.get(h, 0)
            first_redempt_day = min_redempt.get(h, 999999)
            has_redeemed = 1 if first_redempt_day < c else 0

            panel_records.append(
                {
                    "household_key": h,
                    "cutoff_day": c,
                    "in_active_campaign_c": in_active,
                    "active_campaign_type_c": camp_type,
                    "n_campaigns_started_c": n_started,
                    "redeemed_coupon_c": has_redeemed,
                }
            )

    res_df = pd.DataFrame(panel_records)
    # Ensure active_campaign_type_c is categorical with consistent ordering
    type_categories = ["Không có", "Type A", "Type B", "Type C", "Nhiều loại"]
    res_df["active_campaign_type_c"] = pd.Categorical(
        res_df["active_campaign_type_c"], categories=type_categories, ordered=False
    )
    return res_df

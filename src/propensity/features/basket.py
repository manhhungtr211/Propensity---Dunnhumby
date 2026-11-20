"""Basket-level aggregations and point-in-time basket size features."""

from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd

from propensity.eda.propensity_rates import asof_aggregate


def basket_table(lines: pd.DataFrame) -> pd.DataFrame:
    """Aggregate enriched transaction lines to basket-level grain.

    Parameters
    ----------
    lines : pd.DataFrame
        Transaction lines containing household_key, BASKET_ID, DAY, SALES_VALUE,
        and optionally is_merch, is_fuel, PRODUCT_ID.

    Returns
    -------
    pd.DataFrame
        One row per basket with columns:
        - household_key
        - BASKET_ID
        - DAY
        - merch_spend
        - n_products (distinct merchandise PRODUCT_ID)
        - has_fuel (boolean)
    """
    req_cols = {"household_key", "BASKET_ID", "DAY", "SALES_VALUE"}
    missing = req_cols - set(lines.columns)
    if missing:
        raise ValueError(f"Bảng lines thiếu các cột: {missing}")

    df = lines.copy()
    if "is_merch" not in df.columns:
        df["is_merch"] = True
    if "is_fuel" not in df.columns:
        df["is_fuel"] = False
    if "PRODUCT_ID" not in df.columns:
        df["PRODUCT_ID"] = 1

    # Filter merchandise lines for spend and product count
    merch_df = df[df["is_merch"]]
    merch_agg = (
        merch_df.groupby(["household_key", "BASKET_ID", "DAY"], as_index=False)
        .agg(
            merch_spend=("SALES_VALUE", "sum"),
            n_products=("PRODUCT_ID", "nunique"),
        )
    )

    # Check fuel per basket
    fuel_agg = (
        df.groupby(["household_key", "BASKET_ID", "DAY"], as_index=False)["is_fuel"]
        .any()
        .rename(columns={"is_fuel": "has_fuel"})
    )

    # Merge together
    baskets = fuel_agg.merge(
        merch_agg,
        on=["household_key", "BASKET_ID", "DAY"],
        how="left",
    )
    baskets["merch_spend"] = baskets["merch_spend"].fillna(0.0).astype(float)
    baskets["n_products"] = baskets["n_products"].fillna(0).astype(int)
    baskets["has_fuel"] = baskets["has_fuel"].fillna(False).astype(bool)

    cols = ["household_key", "BASKET_ID", "DAY", "merch_spend", "n_products", "has_fuel"]
    return baskets[cols].sort_values(["DAY", "household_key", "BASKET_ID"]).reset_index(drop=True)


def day_basket_table(
    baskets: pd.DataFrame,
    lines: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate basket-level data to household x DAY grain.

    Parameters
    ----------
    baskets : pd.DataFrame
        Output of basket_table containing household_key, DAY, merch_spend, n_products.
    lines : pd.DataFrame, optional
        Raw lines to compute exact distinct PRODUCT_ID across the entire day if multiple baskets exist.

    Returns
    -------
    pd.DataFrame
        One row per household x DAY with columns:
        - household_key
        - DAY
        - day_spend
        - day_products
    """
    req_cols = {"household_key", "DAY", "merch_spend", "n_products"}
    missing = req_cols - set(baskets.columns)
    if missing:
        raise ValueError(f"Bảng baskets thiếu các cột: {missing}")

    if lines is not None and "PRODUCT_ID" in lines.columns:
        is_merch_col = "is_merch" if "is_merch" in lines.columns else None
        merch_lines = lines[lines[is_merch_col]] if is_merch_col else lines
        prod_day = (
            merch_lines.groupby(["household_key", "DAY"], as_index=False)["PRODUCT_ID"]
            .nunique()
            .rename(columns={"PRODUCT_ID": "day_products"})
        )
        spend_day = (
            baskets.groupby(["household_key", "DAY"], as_index=False)["merch_spend"]
            .sum()
            .rename(columns={"merch_spend": "day_spend"})
        )
        day_tbl = spend_day.merge(prod_day, on=["household_key", "DAY"], how="left")
        day_tbl["day_products"] = day_tbl["day_products"].fillna(0).astype(int)
    else:
        day_tbl = (
            baskets.groupby(["household_key", "DAY"], as_index=False)
            .agg(
                day_spend=("merch_spend", "sum"),
                day_products=("n_products", "sum"),
            )
        )

    return day_tbl.sort_values(["household_key", "DAY"]).reset_index(drop=True)


def basket_size_bands(baskets: pd.DataFrame) -> pd.DataFrame:
    """Compute basket size distribution across 5 canonical product-count bands.

    Bands:
    1–2, 3–5, 6–10, 11–20, 21+ sản phẩm.
    Only merchandise baskets (n_products >= 1) are included.

    Returns
    -------
    pd.DataFrame
        Columns: band, n_baskets, pct_baskets, total_spend, pct_spend, median_spend.
    """
    merch_baskets = baskets[baskets["n_products"] >= 1].copy()
    if merch_baskets.empty:
        return pd.DataFrame(
            columns=["band", "n_baskets", "pct_baskets", "total_spend", "pct_spend", "median_spend"]
        )

    bins = [0, 2, 5, 10, 20, np.inf]
    labels = ["1–2", "3–5", "6–10", "11–20", "21+"]

    merch_baskets["band"] = pd.cut(
        merch_baskets["n_products"],
        bins=bins,
        labels=labels,
        right=True,
    )

    total_baskets = len(merch_baskets)
    total_spend_all = merch_baskets["merch_spend"].sum()

    grouped = (
        merch_baskets.groupby("band", observed=False)
        .agg(
            n_baskets=("BASKET_ID", "count"),
            total_spend=("merch_spend", "sum"),
            median_spend=("merch_spend", "median"),
        )
        .reset_index()
    )

    grouped["pct_baskets"] = grouped["n_baskets"] / total_baskets
    grouped["pct_spend"] = grouped["total_spend"] / total_spend_all if total_spend_all > 0 else 0.0
    grouped["band"] = pd.Categorical(grouped["band"], categories=labels, ordered=True)
    grouped = grouped.sort_values("band").reset_index(drop=True)

    cols = ["band", "n_baskets", "pct_baskets", "total_spend", "pct_spend", "median_spend"]
    return grouped[cols]


def basket_features(
    day_tbl: pd.DataFrame,
    cutoffs: Sequence[int],
) -> pd.DataFrame:
    """Compute point-in-time basket features strictly as-of cutoff (DAY < c).

    Features:
    - median_products_per_day_c: Median số sản phẩm/ngày mua trước cutoff
    - last_day_products_c: Số sản phẩm ngày mua gần nhất trước cutoff
    - last_basket_ratio_c: last_day_products_c / median_products_per_day_c (null nếu <3 ngày mua)
    - last_spend_ratio_c: last_day_spend_c / median_spend_per_day_c (null nếu <3 ngày mua)

    Parameters
    ----------
    day_tbl : pd.DataFrame
        Output of day_basket_table containing household_key, DAY, day_spend, day_products.
    cutoffs : Sequence[int]
        List of cutoff days.

    Returns
    -------
    pd.DataFrame
        One row per (household_key, cutoff_day) with features.
    """
    req_cols = {"household_key", "DAY", "day_spend", "day_products"}
    missing = req_cols - set(day_tbl.columns)
    if missing:
        raise ValueError(f"Bảng day_tbl thiếu các cột: {missing}")

    def _agg_asof(sub: pd.DataFrame) -> pd.DataFrame:
        if sub.empty:
            return pd.DataFrame(
                columns=[
                    "household_key",
                    "median_products_per_day_c",
                    "last_day_products_c",
                    "last_basket_ratio_c",
                    "last_spend_ratio_c",
                ]
            )

        # Count purchase days, median products, median spend
        n_days = sub.groupby("household_key")["DAY"].count()
        med_prods = sub.groupby("household_key")["day_products"].median()
        med_spend = sub.groupby("household_key")["day_spend"].median()

        # Last purchase day row
        sorted_sub = sub.sort_values("DAY")
        last_rows = sorted_sub.groupby("household_key").last()
        last_prods = last_rows["day_products"]
        last_spend = last_rows["day_spend"]

        res = pd.DataFrame(
            {
                "household_key": n_days.index,
                "n_purchase_days": n_days.values,
                "median_products_per_day_c": med_prods.values,
                "last_day_products_c": last_prods.values,
                "median_spend_per_day_c": med_spend.values,
                "last_day_spend_c": last_spend.values,
            }
        )

        # Ratios are valid only when household has >= 3 distinct purchase days
        has_3_days = res["n_purchase_days"] >= 3
        res["last_basket_ratio_c"] = np.where(
            has_3_days & (res["median_products_per_day_c"] > 0),
            res["last_day_products_c"] / res["median_products_per_day_c"],
            np.nan,
        )
        res["last_spend_ratio_c"] = np.where(
            has_3_days & (res["median_spend_per_day_c"] > 0),
            res["last_day_spend_c"] / res["median_spend_per_day_c"],
            np.nan,
        )

        cols = [
            "household_key",
            "median_products_per_day_c",
            "last_day_products_c",
            "last_basket_ratio_c",
            "last_spend_ratio_c",
        ]
        return res[cols]

    return asof_aggregate(day_tbl, cutoffs=cutoffs, agg_fn=_agg_asof)

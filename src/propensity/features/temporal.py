"""Temporal purchase dynamics, trend, lapse ratios, and weekly cutoff activity."""

from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd

from propensity.eda.propensity_rates import asof_aggregate


def temporal_features(
    purchase_days: pd.DataFrame,
    cutoffs: Sequence[int],
) -> pd.DataFrame:
    """Calculate point-in-time temporal acceleration/slowdown strictly as-of cutoff (DAY < c).

    Feature:
    - recent_ipt_ratio_c: mean(last 3 gaps) / median(prior gaps)
      Null if the household has fewer than 8 gaps (< 9 distinct purchase days).

    Parameters
    ----------
    purchase_days : pd.DataFrame
        Table with household_key and DAY (distinct purchase days).
    cutoffs : Sequence[int]
        List of cutoff days.

    Returns
    -------
    pd.DataFrame
        One row per (household_key, cutoff_day) with recent_ipt_ratio_c.
    """
    req_cols = {"household_key", "DAY"}
    missing = req_cols - set(purchase_days.columns)
    if missing:
        raise ValueError(f"Bảng purchase_days thiếu các cột: {missing}")

    # Ensure unique (household_key, DAY)
    pdays = purchase_days[["household_key", "DAY"]].drop_duplicates().sort_values(["household_key", "DAY"])

    def _agg_asof(sub: pd.DataFrame) -> pd.DataFrame:
        if sub.empty:
            return pd.DataFrame(columns=["household_key", "recent_ipt_ratio_c"])

        # Group days by household
        hh_days = sub.groupby("household_key")["DAY"].apply(lambda s: s.values)

        hh_keys = []
        ratios = []

        for hh, days in hh_days.items():
            hh_keys.append(hh)
            if len(days) < 9:  # fewer than 8 gaps
                ratios.append(np.nan)
            else:
                gaps = np.diff(days)
                recent_mean = float(np.mean(gaps[-3:]))
                prior_med = float(np.median(gaps[:-3]))
                if prior_med > 0:
                    ratios.append(recent_mean / prior_med)
                else:
                    ratios.append(np.nan)

        return pd.DataFrame(
            {
                "household_key": hh_keys,
                "recent_ipt_ratio_c": ratios,
            }
        )

    return asof_aggregate(pdays, cutoffs=cutoffs, agg_fn=_agg_asof)


def trend_bin(s: pd.Series) -> pd.Series:
    """Bin recent_ipt_ratio into 4 trend categories.

    Bins:
    - < 0.75: Tăng tốc (gaps gần đây ngắn hơn bình thường)
    - 0.75–1.25: Ổn định
    - 1.25–2.0: Chậm lại
    - > 2.0: Chậm hẳn
    - Null / NaN nếu chưa đủ 8 gap.
    """
    bins = [-np.inf, 0.75, 1.25, 2.0, np.inf]
    labels = ["< 0.75 (Tăng tốc)", "0.75–1.25 (Ổn định)", "1.25–2.0 (Chậm lại)", "> 2.0 (Chậm hẳn)"]

    return pd.cut(
        s,
        bins=bins,
        labels=labels,
        right=False,  # [0.75, 1.25)
    )


def lapse_bin(s: pd.Series) -> pd.Series:
    """Bin lapse_ratio (recency / median_ipt) into 4 relative lapse categories.

    Bins:
    - ≤ 1: Đúng hạn / Chưa tới chu kỳ
    - 1–2: Hơi trễ (1-2x median IPT)
    - 2–4: Trễ vừa (2-4x median IPT)
    - > 4: Trễ nhiều (>4x median IPT)
    - Null / NaN nếu median_ipt là null.
    """
    bins = [-np.inf, 1.0, 2.0, 4.0, np.inf]
    labels = ["≤ 1", "1–2", "2–4", "> 4"]

    return pd.cut(
        s,
        bins=bins,
        labels=labels,
        right=True,  # (1, 2]
    )


def weekly_activity(lines: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction activity by calendar week (WEEK_NO).

    Parameters
    ----------
    lines : pd.DataFrame
        Transaction lines containing WEEK_NO, household_key, BASKET_ID, DAY.

    Returns
    -------
    pd.DataFrame
        Columns: WEEK_NO, active_hh, n_baskets, baskets_per_active_hh, min_day, max_day.
    """
    req_cols = {"WEEK_NO", "household_key", "BASKET_ID", "DAY"}
    missing = req_cols - set(lines.columns)
    if missing:
        raise ValueError(f"Bảng lines thiếu các cột: {missing}")

    grouped = (
        lines.groupby("WEEK_NO", as_index=False)
        .agg(
            active_hh=("household_key", "nunique"),
            n_baskets=("BASKET_ID", "nunique"),
            min_day=("DAY", "min"),
            max_day=("DAY", "max"),
        )
        .sort_values("WEEK_NO")
        .reset_index(drop=True)
    )

    grouped["baskets_per_active_hh"] = grouped["n_baskets"] / grouped["active_hh"]
    cols = ["WEEK_NO", "active_hh", "n_baskets", "baskets_per_active_hh", "min_day", "max_day"]
    return grouped[cols]


def cutoff_return_rates(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute observed 7-day return rates and detect unusual cutoffs using robust MAD z-scores.

    Parameters
    ----------
    panel : pd.DataFrame
        Household-cutoff panel containing cutoff_day and target.

    Returns
    -------
    pd.DataFrame
        Columns: cutoff_day, n_eligible, n_positive, return_rate, robust_z, is_unusual.
    """
    req_cols = {"cutoff_day", "target"}
    missing = req_cols - set(panel.columns)
    if missing:
        raise ValueError(f"Bảng panel thiếu các cột: {missing}")

    grouped = (
        panel.groupby("cutoff_day", as_index=False)
        .agg(
            n_eligible=("target", "count"),
            n_positive=("target", "sum"),
        )
        .sort_values("cutoff_day")
        .reset_index(drop=True)
    )

    grouped["return_rate"] = grouped["n_positive"] / grouped["n_eligible"]

    med_rate = float(np.median(grouped["return_rate"]))
    mad = float(np.median(np.abs(grouped["return_rate"] - med_rate)))
    scale = 1.4826 * mad

    if scale > 0:
        grouped["robust_z"] = (grouped["return_rate"] - med_rate) / scale
    else:
        grouped["robust_z"] = 0.0

    grouped["is_unusual"] = np.abs(grouped["robust_z"]) > 3.5

    cols = ["cutoff_day", "n_eligible", "n_positive", "return_rate", "robust_z", "is_unusual"]
    return grouped[cols]

"""Customer deep-dive diagnostics: Q-F continuity, Q-CV regularity, Q-M spend decomposition, and Q-IPT profiling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd

from propensity.eda.customer_overview import (
    CustomerOverviewError,
    build_household_summary,
    build_purchase_days,
)


class CustomerDeepDiveError(ValueError):
    """Raised when customer deep dive operations encounter invalid data or schema."""


@dataclass(frozen=True)
class DeepDiveThresholds:
    """Configurable thresholds for customer deep dive diagnostics."""

    high_f_quantile: float = 0.80      # F cao = top 20%
    continuous_coverage: float = 0.75  # >= 75% số tuần có mua
    burst_window_days: int = 91        # cửa sổ 13 tuần
    burst_share: float = 0.50          # >= 50% ngày mua dồn trong 1 cửa sổ
    min_gaps: int = 10                 # dưới 10 khoảng cách thì CV không tin cậy
    long_break_k: float = 2.0          # nghỉ dài = gap > 2 * median IPT của chính household
    cv_high: float = 1.0               # ngưỡng phân loại CV cao
    ipt_bands: tuple[int, int] = (7, 14)  # <=7 | 8–14 | >14
    f_share_redundant: float = 0.60    # Q-M: ngưỡng tỷ lệ đóng góp của F coi là thừa thông tin


@dataclass(frozen=True, slots=True)
class SpendDecompositionResult:
    """Results of log-variance decomposition of spend per week."""

    variance_shares: pd.DataFrame
    f_share: float
    tpd_share: float
    spt_share: float
    is_f_redundant: bool
    decision_rule: str


@dataclass(frozen=True, slots=True)
class CustomerDeepDiveReport:
    """Consolidated report containing all customer deep dive diagnostic tables."""

    purchase_days: pd.DataFrame
    continuity: pd.DataFrame
    high_f_classified: pd.DataFrame
    high_f_summary: pd.DataFrame
    cv_diagnostics: pd.DataFrame
    cv_driver_summary: pd.DataFrame
    spend_decomposition: SpendDecompositionResult
    ipt_band_profile: pd.DataFrame
    ipt_gt14_subtypes: pd.DataFrame
    thresholds: DeepDiveThresholds


def activity_continuity(
    purchase_days: pd.DataFrame,
    summary: pd.DataFrame,
    thr: DeepDiveThresholds = DeepDiveThresholds(),
) -> pd.DataFrame:
    """Calculate weekly purchase activity continuity and burstiness metrics per household.

    Parameters
    ----------
    purchase_days : pd.DataFrame
        Output of build_purchase_days (household_key, DAY, n_trips_day, spend_day, gap).
    summary : pd.DataFrame
        Output of build_household_summary.
    thr : DeepDiveThresholds
        Configuration thresholds.

    Returns
    -------
    pd.DataFrame
        Table with one row per household:
        - household_key
        - first_day, last_day, span_days, n_purchase_days, F
        - n_span_weeks: span in weeks relative to first_day
        - n_active_weeks: count of distinct relative weeks with >= 1 purchase
        - weekly_coverage: n_active_weeks / n_span_weeks
        - max_window_share: maximum share of purchase days inside any 91-day window
    """
    if purchase_days.empty or summary.empty:
        raise CustomerDeepDiveError("Dữ liệu purchase_days hoặc summary rỗng")

    # Map first_day to purchase_days for relative week indexing
    first_day_map = summary.set_index("household_key")["first_day"].to_dict()
    pdays = purchase_days.copy()
    pdays["first_day"] = pdays["household_key"].map(first_day_map)
    pdays["rel_week"] = (pdays["DAY"] - pdays["first_day"]) // 7

    active_weeks_series = pdays.groupby("household_key")["rel_week"].nunique()

    # Calculate max_window_share per household in a burst_window_days sliding window
    def _calc_max_share(days: np.ndarray, window: int) -> float:
        if len(days) == 0:
            return 0.0
        if days[-1] - days[0] + 1 <= window:
            return 1.0
        right_idx = np.searchsorted(days, days + window - 1, side="right")
        counts_in_win = right_idx - np.arange(len(days))
        return float(np.max(counts_in_win) / len(days))

    max_shares_series = pdays.groupby("household_key")["DAY"].apply(
        lambda s: _calc_max_share(s.values, thr.burst_window_days)
    )

    base = summary[[
        "household_key",
        "first_day",
        "last_day",
        "span_days",
        "n_purchase_days",
        "F",
    ]].copy()

    base["n_span_weeks"] = ((base["span_days"] - 1) // 7 + 1).astype(int)
    base["n_active_weeks"] = base["household_key"].map(active_weeks_series).fillna(0).astype(int)
    base["weekly_coverage"] = np.where(
        base["n_span_weeks"] > 0,
        base["n_active_weeks"] / base["n_span_weeks"],
        0.0,
    )
    base["max_window_share"] = base["household_key"].map(max_shares_series).fillna(0.0)

    return base.sort_values("household_key").reset_index(drop=True)


def classify_high_f(
    continuity: pd.DataFrame,
    thr: DeepDiveThresholds = DeepDiveThresholds(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify high-F households into 'liên tục', 'mua dồn', or 'trung gian'.

    Parameters
    ----------
    continuity : pd.DataFrame
        Output of activity_continuity.
    thr : DeepDiveThresholds
        Configuration thresholds.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (high_f_households, f_pattern_summary)
    """
    if continuity.empty:
        raise CustomerDeepDiveError("Bảng continuity rỗng")

    q_high_f = continuity["F"].quantile(thr.high_f_quantile)
    high_f = continuity[continuity["F"] >= q_high_f].copy()

    def _assign_pattern(row: pd.Series) -> str:
        if row["span_days"] < thr.burst_window_days or row["max_window_share"] >= thr.burst_share:
            return "mua dồn"
        elif row["weekly_coverage"] >= thr.continuous_coverage:
            return "liên tục"
        else:
            return "trung gian"

    high_f["f_pattern"] = high_f.apply(_assign_pattern, axis=1)

    # Summary table
    order = ["liên tục", "trung gian", "mua dồn"]
    total_high_f = len(high_f)
    records = []
    for pat in order:
        sub = high_f[high_f["f_pattern"] == pat]
        n_hh = len(sub)
        pct = (n_hh / total_high_f) * 100.0 if total_high_f > 0 else 0.0
        records.append({
            "f_pattern": pat,
            "n_households": n_hh,
            "pct_households": round(pct, 1),
            "median_f": round(sub["F"].median(), 2) if n_hh > 0 else np.nan,
            "median_weekly_coverage": round(sub["weekly_coverage"].median(), 3) if n_hh > 0 else np.nan,
            "median_max_window_share": round(sub["max_window_share"].median(), 3) if n_hh > 0 else np.nan,
            "median_span_days": round(sub["span_days"].median(), 1) if n_hh > 0 else np.nan,
            "median_n_purchase_days": round(sub["n_purchase_days"].median(), 1) if n_hh > 0 else np.nan,
        })

    summary_df = pd.DataFrame(records)
    return high_f.sort_values("household_key").reset_index(drop=True), summary_df


def cv_diagnostics(
    purchase_days: pd.DataFrame,
    summary: pd.DataFrame,
    thr: DeepDiveThresholds = DeepDiveThresholds(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diagnose causes of high CV (irregularity vs sample size vs long breaks).

    Parameters
    ----------
    purchase_days : pd.DataFrame
        Output of build_purchase_days.
    summary : pd.DataFrame
        Output of build_household_summary.
    thr : DeepDiveThresholds
        Configuration thresholds.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (diagnostics_df, cv_driver_summary)
    """
    if purchase_days.empty or summary.empty:
        raise CustomerDeepDiveError("Dữ liệu purchase_days hoặc summary rỗng")

    summary_map = summary.set_index("household_key")
    records = []

    for hh_key, grp in purchase_days.groupby("household_key"):
        gaps = grp["gap"].dropna().values
        n_gaps = len(gaps)
        med_ipt = summary_map.loc[hh_key, "median_ipt"]
        cv_ipt = summary_map.loc[hh_key, "cv_ipt"]

        if n_gaps == 0:
            records.append({
                "household_key": hh_key,
                "n_gaps": 0,
                "max_gap": np.nan,
                "n_long_breaks": 0,
                "cv_drop_max": np.nan,
                "robust_cv": np.nan,
                "cv_ipt": cv_ipt,
            })
            continue

        max_g = float(np.max(gaps))
        # Long break: gap > long_break_k * median_ipt
        if pd.notna(med_ipt) and med_ipt > 0:
            n_long = int(np.sum(gaps > (thr.long_break_k * med_ipt)))
        else:
            n_long = 0

        # cv_drop_max: drop one occurrence of the maximum gap
        if n_gaps >= 3:
            max_idx = int(np.argmax(gaps))
            reduced = np.delete(gaps, max_idx)
            m_red = np.mean(reduced)
            s_red = np.std(reduced, ddof=1)
            cv_drop = float(s_red / m_red) if m_red > 0 else np.nan
        else:
            cv_drop = np.nan

        # robust_cv: IQR / median
        if n_gaps >= 2 and pd.notna(med_ipt) and med_ipt > 0:
            q75, q25 = np.percentile(gaps, [75, 25])
            rob_cv = float((q75 - q25) / med_ipt)
        else:
            rob_cv = np.nan

        records.append({
            "household_key": hh_key,
            "n_gaps": n_gaps,
            "max_gap": max_g,
            "n_long_breaks": n_long,
            "cv_drop_max": cv_drop,
            "robust_cv": rob_cv,
            "cv_ipt": cv_ipt,
        })

    diag_df = pd.DataFrame(records)

    def _assign_cv_driver(row: pd.Series) -> str:
        cv = row["cv_ipt"]
        if pd.isna(cv):
            return "không đủ dữ liệu"
        if cv <= thr.cv_high:
            return "đều đặn"

        # If CV > cv_high (1.0), evaluate in prioritized sequence:
        # 1. ít lần mua
        if row["n_gaps"] < thr.min_gaps:
            return "ít lần mua"
        # 2. do nghỉ dài
        if (pd.notna(row["cv_drop_max"]) and row["cv_drop_max"] <= thr.cv_high) or (
            row["n_long_breaks"] > 0 and pd.notna(row["robust_cv"]) and row["robust_cv"] <= 0.5
        ):
            return "do nghỉ dài"
        # 3. thất thường thật
        return "thất thường thật"

    diag_df["cv_driver"] = diag_df.apply(_assign_cv_driver, axis=1)

    # Summary table for high-CV households (CV > cv_high)
    high_cv = diag_df[diag_df["cv_ipt"] > thr.cv_high]
    total_high_cv = len(high_cv)
    order = ["thất thường thật", "do nghỉ dài", "ít lần mua"]

    summary_records = []
    for drv in order:
        sub = high_cv[high_cv["cv_driver"] == drv]
        n_hh = len(sub)
        pct = (n_hh / total_high_cv) * 100.0 if total_high_cv > 0 else 0.0
        summary_records.append({
            "cv_driver": drv,
            "n_households": n_hh,
            "pct_share": round(pct, 1),
            "median_cv_ipt": round(sub["cv_ipt"].median(), 2) if n_hh > 0 else np.nan,
            "median_cv_drop_max": round(sub["cv_drop_max"].median(), 2) if n_hh > 0 else np.nan,
            "median_robust_cv": round(sub["robust_cv"].median(), 2) if n_hh > 0 else np.nan,
            "median_n_gaps": round(sub["n_gaps"].median(), 1) if n_hh > 0 else np.nan,
        })

    summary_df = pd.DataFrame(summary_records)
    return diag_df.sort_values("household_key").reset_index(drop=True), summary_df


def spend_decomposition(
    summary: pd.DataFrame,
    exclude_short_span: bool = True,
    thr: DeepDiveThresholds = DeepDiveThresholds(),
) -> SpendDecompositionResult:
    """Decompose log-variance of spend per week into F, trips/day, and spend/trip.

    Mathematical identity:
        spend_per_week = F * (n_trips / n_purchase_days) * spend_per_trip
    Log transformation:
        ln(spend_per_week) = ln(F) + ln(trips_per_day) + ln(spend_per_trip)
    Variance decomposition:
        var(ln Y) = cov(ln F, ln Y) + cov(ln tpd, ln Y) + cov(ln spt, ln Y)
        share_k = cov(ln X_k, ln Y) / var(ln Y), where sum(share_k) == 1.0

    Parameters
    ----------
    summary : pd.DataFrame
        Output of build_household_summary.
    exclude_short_span : bool
        Whether to exclude households with span_days < 28 (default True).
    thr : DeepDiveThresholds
        Configuration thresholds.

    Returns
    -------
    SpendDecompositionResult
        Dataclass containing variance share table, component shares, and decision rule.
    """
    if summary.empty:
        raise CustomerDeepDiveError("Bảng summary rỗng")

    s = summary.copy()
    if exclude_short_span and "short_span" in s.columns:
        s = s[~s["short_span"]].copy()

    s = s[(s["spend_per_week"] > 0) & (s["F"] > 0) & (s["spend_per_trip"] > 0)].copy()
    s["trips_per_day"] = s["n_trips"] / s["n_purchase_days"]
    s = s[s["trips_per_day"] > 0].copy()

    if len(s) < 2:
        raise CustomerDeepDiveError("Cần ít nhất 2 hộ gia đình hợp lệ để phân rã phương sai")

    log_y = np.log(s["spend_per_week"].values)
    log_f = np.log(s["F"].values)
    log_tpd = np.log(s["trips_per_day"].values)
    log_spt = np.log(s["spend_per_trip"].values)

    var_y = float(np.var(log_y, ddof=1))
    if np.isnan(var_y) or var_y <= 0:
        raise CustomerDeepDiveError("Phương sai của ln(spend_per_week) không hợp lệ hoặc bằng 0")

    cov_f = float(np.cov(log_f, log_y)[0, 1])
    cov_tpd = float(np.cov(log_tpd, log_y)[0, 1])
    cov_spt = float(np.cov(log_spt, log_y)[0, 1])

    f_share = cov_f / var_y
    tpd_share = cov_tpd / var_y
    spt_share = cov_spt / var_y

    # Spearman rank correlations with spend_per_week
    spend_rank = s["spend_per_week"].rank()
    sp_f = float(s["F"].rank().corr(spend_rank))
    sp_tpd = float(s["trips_per_day"].rank().corr(spend_rank))
    sp_spt = float(s["spend_per_trip"].rank().corr(spend_rank))

    shares_df = pd.DataFrame([
        {
            "component": "F (Số ngày mua/tuần)",
            "variance_share": round(f_share, 4),
            "variance_share_pct": round(f_share * 100.0, 2),
            "spearman_corr_with_spend": round(float(sp_f), 3),
        },
        {
            "component": "Spend per trip (Chi tiêu/chuyến)",
            "variance_share": round(spt_share, 4),
            "variance_share_pct": round(spt_share * 100.0, 2),
            "spearman_corr_with_spend": round(float(sp_spt), 3),
        },
        {
            "component": "Trips per day (Số trip/ngày mua)",
            "variance_share": round(tpd_share, 4),
            "variance_share_pct": round(tpd_share * 100.0, 2),
            "spearman_corr_with_spend": round(float(sp_tpd), 3),
        },
    ])

    is_redundant = f_share >= thr.f_share_redundant
    if is_redundant:
        rule = (
            f"Tỷ lệ đóng góp của F vào phương sai ln(spend/tuần) đạt {f_share*100:.1f}% "
            f"(>= ngưỡng {thr.f_share_redundant*100:.0f}%). Biến spend_per_week thừa thông tin so với F. "
            "Quyết định Feature Modeling: Loại bỏ spend_per_week, chỉ giữ lại F và spend_per_trip để tránh đa cộng tuyến."
        )
    else:
        rule = (
            f"Tỷ lệ đóng góp của F vào phương sai ln(spend/tuần) là {f_share*100:.1f}% "
            f"(< ngưỡng {thr.f_share_redundant*100:.0f}%). Biến spend_per_week mang thông tin độc lập đáng kể."
        )

    return SpendDecompositionResult(
        variance_shares=shares_df,
        f_share=f_share,
        tpd_share=tpd_share,
        spt_share=spt_share,
        is_f_redundant=is_redundant,
        decision_rule=rule,
    )


def long_ipt_profile(
    summary: pd.DataFrame,
    continuity: pd.DataFrame,
    cv_diag: pd.DataFrame,
    thr: DeepDiveThresholds = DeepDiveThresholds(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profile customer behavior across IPT bands and classify > 14 day subtypes.

    Parameters
    ----------
    summary : pd.DataFrame
        Output of build_household_summary.
    continuity : pd.DataFrame
        Output of activity_continuity.
    cv_diag : pd.DataFrame
        Output of cv_diagnostics (diagnostics_df).
    thr : DeepDiveThresholds
        Configuration thresholds.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (ipt_band_profile, ipt_gt14_subtypes)
    """
    if summary.empty:
        raise CustomerDeepDiveError("Bảng summary rỗng")

    # Merge auxiliary metrics
    merged = summary.copy()
    merged["trips_per_day"] = merged["n_trips"] / merged["n_purchase_days"]

    if "weekly_coverage" in continuity.columns:
        cov_map = continuity.set_index("household_key")["weekly_coverage"].to_dict()
        merged["weekly_coverage"] = merged["household_key"].map(cov_map)
    else:
        merged["weekly_coverage"] = np.nan

    if "n_long_breaks" in cv_diag.columns:
        lb_map = cv_diag.set_index("household_key")["n_long_breaks"].to_dict()
        merged["n_long_breaks"] = merged["household_key"].map(lb_map)
    else:
        merged["n_long_breaks"] = 0

    # Assign IPT bands
    b1, b2 = thr.ipt_bands
    def _get_band(ipt: float) -> str:
        if pd.isna(ipt):
            return "null"
        elif ipt <= b1:
            return f"≤{b1}"
        elif ipt <= b2:
            return f"{b1+1}–{b2}"
        else:
            return f">{b2}"

    merged["ipt_band"] = merged["median_ipt"].apply(_get_band)

    # Table 1: IPT Band Profile
    total_hh = len(merged)
    total_spend_all = merged["total_spend"].sum()
    band_order = [f"≤{b1}", f"{b1+1}–{b2}", f">{b2}", "null"]

    band_records = []
    for b in band_order:
        sub = merged[merged["ipt_band"] == b]
        n_b = len(sub)
        if n_b == 0:
            continue
        pct_hh = (n_b / total_hh) * 100.0
        pct_sp = (sub["total_spend"].sum() / total_spend_all) * 100.0 if total_spend_all > 0 else 0.0

        band_records.append({
            "ipt_band": b,
            "n_households": n_b,
            "pct_households": round(pct_hh, 1),
            "pct_total_spend": round(pct_sp, 1),
            "median_purchase_days": round(sub["n_purchase_days"].median(), 1),
            "median_span_days": round(sub["span_days"].median(), 1),
            "pct_short_span": round(sub["short_span"].mean() * 100.0, 1),
            "median_f": round(sub["F"].median(), 2),
            "median_trips_per_day": round(sub["trips_per_day"].median(), 2),
            "median_spend_per_trip": round(sub["spend_per_trip"].median(), 2),
            "median_cv_ipt": round(sub["cv_ipt"].median(), 2) if sub["cv_ipt"].notna().any() else np.nan,
            "median_n_long_breaks": round(sub["n_long_breaks"].median(), 1),
            "median_weekly_coverage": round(sub["weekly_coverage"].median(), 3) if sub["weekly_coverage"].notna().any() else np.nan,
            "median_first_day": round(sub["first_day"].median(), 1),
            "median_last_day": round(sub["last_day"].median(), 1),
        })

    band_profile_df = pd.DataFrame(band_records)

    # Table 2: Subtypes for > 14 band
    gt14_band_name = f">{b2}"
    gt14 = merged[merged["ipt_band"] == gt14_band_name].copy()

    p25_spend = summary["spend_per_trip"].quantile(0.25)
    p75_spend = summary["spend_per_trip"].quantile(0.75)

    def _assign_subtype(r: pd.Series) -> str:
        # a. Lịch sử ngắn
        if r["n_purchase_days"] < 10 or r["short_span"]:
            return "lịch sử ngắn"
        # b. Mua dự trữ
        elif r["spend_per_trip"] >= p75_spend:
            return "mua dự trữ"
        # c. Mua bổ sung
        elif r["spend_per_trip"] <= p25_spend:
            return "mua bổ sung"
        # d. Nhịp dài đều đặn
        elif pd.notna(r["cv_ipt"]) and r["cv_ipt"] < 0.5:
            return "nhịp dài đều đặn"
        # e. Khác
        else:
            return "khác"

    gt14["ipt_subtype"] = gt14.apply(_assign_subtype, axis=1)

    subtype_order = ["lịch sử ngắn", "mua dự trữ", "mua bổ sung", "nhịp dài đều đặn", "khác"]
    n_gt14 = len(gt14)
    subtype_records = []
    for st in subtype_order:
        sub = gt14[gt14["ipt_subtype"] == st]
        n_st = len(sub)
        pct_st = (n_st / n_gt14) * 100.0 if n_gt14 > 0 else 0.0
        subtype_records.append({
            "ipt_subtype": st,
            "n_households": n_st,
            "pct_subtypes": round(pct_st, 1),
            "median_spend_per_trip": round(sub["spend_per_trip"].median(), 2) if n_st > 0 else np.nan,
            "median_f": round(sub["F"].median(), 2) if n_st > 0 else np.nan,
            "median_cv_ipt": round(sub["cv_ipt"].median(), 2) if n_st > 0 and sub["cv_ipt"].notna().any() else np.nan,
            "median_n_purchase_days": round(sub["n_purchase_days"].median(), 1) if n_st > 0 else np.nan,
            "median_span_days": round(sub["span_days"].median(), 1) if n_st > 0 else np.nan,
        })

    subtypes_df = pd.DataFrame(subtype_records)
    return band_profile_df, subtypes_df


def build_customer_deep_dive_report(
    transactions: pd.DataFrame,
    thr: DeepDiveThresholds = DeepDiveThresholds(),
) -> CustomerDeepDiveReport:
    """Execute end-to-end customer deep dive pipeline and return CustomerDeepDiveReport.

    Parameters
    ----------
    transactions : pd.DataFrame
        Raw transaction log containing household_key, BASKET_ID, DAY, and SALES_VALUE.
    thr : DeepDiveThresholds
        Configuration thresholds.

    Returns
    -------
    CustomerDeepDiveReport
        Dataclass containing all deep dive tables and diagnostics.
    """
    pdays = build_purchase_days(transactions)
    summary = build_household_summary(transactions)

    continuity = activity_continuity(pdays, summary, thr)
    high_f_cls, high_f_sum = classify_high_f(continuity, thr)
    cv_diag, cv_sum = cv_diagnostics(pdays, summary, thr)
    try:
        spend_decomp = spend_decomposition(summary, exclude_short_span=True, thr=thr)
    except CustomerDeepDiveError:
        spend_decomp = spend_decomposition(summary, exclude_short_span=False, thr=thr)
    ipt_bands, ipt_subtypes = long_ipt_profile(summary, continuity, cv_diag, thr)

    return CustomerDeepDiveReport(
        purchase_days=pdays,
        continuity=continuity,
        high_f_classified=high_f_cls,
        high_f_summary=high_f_sum,
        cv_diagnostics=cv_diag,
        cv_driver_summary=cv_sum,
        spend_decomposition=spend_decomp,
        ipt_band_profile=ipt_bands,
        ipt_gt14_subtypes=ipt_subtypes,
        thresholds=thr,
    )

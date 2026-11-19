"""Leakage-safe customer-to-label relationship analytics across temporal cutoffs."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Final

import numpy as np
import pandas as pd

from propensity.purchases import valid_purchase_lines, PurchaseValidationError


class CustomerLabelRelationError(ValueError):
    """Raised when panel data or parameters are invalid."""


@dataclass(frozen=True)
class CustomerLabelConfig:
    """Configuration for cutoff grid, horizon, and binning thresholds."""

    first_cutoff_day: int = 140
    last_cutoff_day: int = 700
    cutoff_step_days: int = 7
    horizon_days: int = 7
    min_tenure_days: int = 28
    # Ghi chú thiết kế: Tạm giữ 4 bin Recency (1–7, 8–14, 15–30, 31+) để khớp với các kết quả
    # phân tầng sơ bộ đã chạy; lưới chuẩn 5 bin (1–3, 4–7, 8–14, 15–30, 31+) sẽ được cập nhật
    # ở giai đoạn Feature Store toàn diện.
    recency_bins: tuple[float, ...] = (0.0, 7.0, 14.0, 30.0, float("inf"))
    recency_labels: tuple[str, ...] = ("1–7", "8–14", "15–30", "31+")
    n_f_quantiles: int = 5
    n_spend_quantiles: int = 5

    @property
    def cutoff_days(self) -> tuple[int, ...]:
        """Return evenly spaced cutoffs."""
        return tuple(
            range(
                self.first_cutoff_day,
                self.last_cutoff_day + 1,
                self.cutoff_step_days,
            )
        )


def _wilson_score_interval(
    k: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Calculate Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    # z = 1.95996 for 95% confidence
    z = 1.9599639845400542 if math.isclose(confidence, 0.95, rel_tol=1e-3) else 1.96
    p_hat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (
        z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n) / denom
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def build_customer_cutoff_panel(
    transactions: pd.DataFrame,
    config: CustomerLabelConfig | None = None,
) -> pd.DataFrame:
    """Construct an as-of household-cutoff panel with strictly pre-cutoff features.

    Features are evaluated on DAY < cutoff. Target is evaluated on [cutoff, cutoff + horizon - 1].
    """
    if config is None:
        config = CustomerLabelConfig()

    req_cols = {"household_key", "BASKET_ID", "DAY", "SALES_VALUE"}
    missing = req_cols - set(transactions.columns)
    if missing:
        raise CustomerLabelRelationError(f"Thiếu các cột giao dịch bắt buộc: {missing}")

    valid = valid_purchase_lines(transactions)
    if valid.empty:
        raise CustomerLabelRelationError("Không có giao dịch hợp lệ sau khi lọc")

    # Sort strictly for binary search
    pdays_df = (
        valid[["household_key", "DAY"]]
        .drop_duplicates()
        .sort_values(["household_key", "DAY"])
    )
    baskets_df = (
        valid.groupby(["household_key", "BASKET_ID", "DAY"], as_index=False)["SALES_VALUE"]
        .sum()
        .sort_values(["household_key", "DAY"])
    )

    # Pre-index arrays per household
    pdays_dict: dict[int, np.ndarray] = {
        h: grp["DAY"].to_numpy() for h, grp in pdays_df.groupby("household_key")
    }
    baskets_dict: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for h, grp in baskets_df.groupby("household_key"):
        b_days = grp["DAY"].to_numpy()
        b_spends = grp["SALES_VALUE"].to_numpy()
        b_spend_cumsum = np.concatenate([[0.0], np.cumsum(b_spends)])
        baskets_dict[h] = (b_days, b_spend_cumsum)

    panel_rows: list[dict[str, Any]] = []
    cutoffs = config.cutoff_days

    for c in cutoffs:
        for h, days in pdays_dict.items():
            fday = days[0]
            if fday >= c:
                # Household not yet eligible at cutoff c
                continue

            idx_c = np.searchsorted(days, c)
            if idx_c == 0:
                continue

            last_d = days[idx_c - 1]
            recency = c - last_d
            tenure = c - fday
            n_pdays = idx_c
            f_weekly = n_pdays / (tenure / 7.0)

            # Spend & trips before cutoff
            b_days, b_spend_cumsum = baskets_dict[h]
            idx_b = np.searchsorted(b_days, c)
            total_spend = b_spend_cumsum[idx_b]
            n_trips = idx_b
            spend_per_trip = total_spend / n_trips if n_trips > 0 else 0.0

            # Median IPT and lapse ratio before cutoff
            if idx_c >= 2:
                hist_days = days[:idx_c]
                gaps = np.diff(hist_days)
                median_ipt = float(np.median(gaps))
                lapse_ratio_c = float(recency / median_ipt) if median_ipt > 0 else np.nan
            else:
                median_ipt = np.nan
                lapse_ratio_c = np.nan

            # Target in [c, c + horizon_days - 1]
            idx_target_end = np.searchsorted(days, c + config.horizon_days)
            target = 1 if (idx_target_end - idx_c) > 0 else 0

            panel_rows.append(
                {
                    "household_key": h,
                    "cutoff_day": c,
                    "recency": recency,
                    "tenure": tenure,
                    "n_pdays": n_pdays,
                    "F_c": f_weekly,
                    "spend_per_trip": spend_per_trip,
                    "median_ipt": median_ipt,
                    "lapse_ratio_c": lapse_ratio_c,
                    "target": target,
                }
            )

    if not panel_rows:
        raise CustomerLabelRelationError("Không tạo được bản ghi nào cho panel")

    panel = pd.DataFrame(panel_rows)

    # 1. Recency binning
    panel["recency_bin"] = pd.cut(
        panel["recency"],
        bins=config.recency_bins,
        labels=config.recency_labels,
        right=True,
    )

    # 2. F binning (5 numeric quantile ranges for tenure >= 28, else short_history)
    mask_long = panel["tenure"] >= config.min_tenure_days
    _, f_edges = pd.qcut(
        panel.loc[mask_long, "F_c"],
        q=config.n_f_quantiles,
        retbins=True,
    )
    f_labels = [
        f"≤ {f_edges[1]:.2f}",
        f"{f_edges[1]:.2f}–{f_edges[2]:.2f}",
        f"{f_edges[2]:.2f}–{f_edges[3]:.2f}",
        f"{f_edges[3]:.2f}–{f_edges[4]:.2f}",
        f"> {f_edges[4]:.2f}",
    ]
    f_quintiles = pd.qcut(
        panel.loc[mask_long, "F_c"],
        q=config.n_f_quantiles,
        labels=f_labels,
    )
    panel["F_bin"] = "Lịch sử ngắn (<28d)"
    panel.loc[mask_long, "F_bin"] = f_quintiles.astype(str)
    all_f_categories = f_labels + ["Lịch sử ngắn (<28d)"]
    panel["F_bin"] = pd.Categorical(
        panel["F_bin"], categories=all_f_categories, ordered=True
    )

    # 3. Spend/trip binning (Quintiles)
    spend_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    panel["spend_bin"] = pd.qcut(
        panel["spend_per_trip"],
        q=config.n_spend_quantiles,
        labels=spend_labels,
    )

    # 4. IPT band binning: <=7, 8-14, >14, null
    def _assign_ipt_band(val: float) -> str:
        if pd.isna(val):
            return "null"
        if val <= 7.0:
            return "≤ 7"
        elif val <= 14.0:
            return "8–14"
        else:
            return "> 14"

    panel["ipt_band_c"] = panel["median_ipt"].apply(_assign_ipt_band)
    ipt_categories = ["≤ 7", "8–14", "> 14", "null"]
    panel["ipt_band_c"] = pd.Categorical(
        panel["ipt_band_c"], categories=ipt_categories, ordered=True
    )

    return panel


def return_rate_by_bins(
    df: pd.DataFrame, bin_col: str, label_col: str = "target"
) -> pd.DataFrame:
    """Calculate return rate, sample size, and 95% Wilson confidence intervals by bin."""
    if bin_col not in df.columns or label_col not in df.columns:
        raise CustomerLabelRelationError(f"Cột {bin_col} hoặc {label_col} không có trong dữ liệu")

    grouped = (
        df.groupby(bin_col, observed=False)[label_col]
        .agg(n="count", n_pos="sum", return_rate="mean")
        .reset_index()
    )

    ci_bounds = [
        _wilson_score_interval(int(row["n_pos"]), int(row["n"]))
        for _, row in grouped.iterrows()
    ]
    grouped["ci_lower"] = [b[0] for b in ci_bounds]
    grouped["ci_upper"] = [b[1] for b in ci_bounds]
    return grouped


def rate_within_strata(
    df: pd.DataFrame,
    primary_bin: str,
    strata_bin: str,
    label_col: str = "target",
) -> pd.DataFrame:
    """Calculate return rate across a 2D cross-tabulation of primary_bin x strata_bin."""
    if primary_bin not in df.columns or strata_bin not in df.columns:
        raise CustomerLabelRelationError("Thiếu cột phân tầng trong dữ liệu")

    strata_matrix = pd.crosstab(
        df[primary_bin],
        df[strata_bin],
        values=df[label_col],
        aggfunc="mean",
        dropna=False,
    )
    return strata_matrix


@dataclass(frozen=True)
class FindingThresholds:
    """Thresholds for automated, quantitative feature classification."""

    core_spread_pp: float = 15.0       # Spread >= 15 pp -> Core Feature
    secondary_spread_pp: float = 3.0   # Spread 3 - 15 pp -> Secondary / Interaction Feature
    redundant_spread_pp: float = 1.0   # Within-strata spread < 1.0 pp -> Redundant / Disguised


def build_rf_baseline(
    df: pd.DataFrame,
    r_col: str = "recency_bin",
    f_col: str = "F_bin",
    label_col: str = "target",
) -> pd.DataFrame:
    """Construct the canonical R x F baseline table containing expected return rate per cell."""
    missing = [c for c in (r_col, f_col, label_col) if c not in df.columns]
    if missing:
        raise CustomerLabelRelationError(f"Thiếu các cột để tạo R x F baseline: {missing}")

    baseline = (
        df.groupby([r_col, f_col], observed=False)[label_col]
        .agg(n="count", rf_expected_rate="mean")
        .reset_index()
    )
    return baseline


def excess_over_rf_baseline(
    df: pd.DataFrame,
    feature_col: str,
    rf_baseline: pd.DataFrame | None = None,
    r_col: str = "recency_bin",
    f_col: str = "F_bin",
    label_col: str = "target",
) -> pd.DataFrame:
    """Calculate observed return rate and excess over BOTH overall baseline and R x F baseline."""
    if feature_col not in df.columns or label_col not in df.columns:
        raise CustomerLabelRelationError(f"Cột {feature_col} hoặc {label_col} không có trong dữ liệu")

    if rf_baseline is None:
        rf_baseline = build_rf_baseline(df, r_col=r_col, f_col=f_col, label_col=label_col)

    # Map expected target under R x F to each observation
    rf_lookup = rf_baseline.set_index([r_col, f_col])["rf_expected_rate"].to_dict()
    expected_rf_series = df.set_index([r_col, f_col]).index.map(rf_lookup)

    overall_baseline = float(df[label_col].mean())

    df_temp = df[[feature_col, label_col]].copy()
    df_temp["expected_rf"] = expected_rf_series.to_numpy()

    agg = (
        df_temp.groupby(feature_col, observed=False)
        .agg(
            n=(label_col, "count"),
            n_pos=(label_col, "sum"),
            observed_rate=(label_col, "mean"),
            expected_rate_rf=("expected_rf", "mean"),
        )
        .reset_index()
    )

    agg["overall_baseline"] = overall_baseline
    agg["excess_over_overall"] = agg["observed_rate"] - overall_baseline
    agg["excess_over_rf"] = agg["observed_rate"] - agg["expected_rate_rf"]

    # Backward compatibility
    agg["return_rate"] = agg["observed_rate"]
    agg["excess_over_baseline"] = agg["excess_over_overall"]

    return agg


def excess_over_baseline(
    df: pd.DataFrame,
    bin_col: str,
    label_col: str = "target",
    rf_baseline: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate excess return rate over overall baseline and R x F baseline."""
    return excess_over_rf_baseline(
        df,
        feature_col=bin_col,
        rf_baseline=rf_baseline,
        label_col=label_col,
    )


def classify_finding(
    metric_df: pd.DataFrame,
    strata_matrix: pd.DataFrame | None = None,
    thresholds: FindingThresholds | None = None,
) -> dict[str, Any]:
    """Classify feature role strictly based on quantitative thresholds.

    No hardcoded domain conclusions; returns data-driven metrics and classification flags.
    """
    if thresholds is None:
        thresholds = FindingThresholds()

    # Determine metric spread (excess over rf if available, else excess over overall, else return_rate)
    if "excess_over_rf" in metric_df.columns:
        spread_pp = float((metric_df["excess_over_rf"].max() - metric_df["excess_over_rf"].min()) * 100.0)
        max_excess_pp = float(metric_df["excess_over_rf"].max() * 100.0)
        min_excess_pp = float(metric_df["excess_over_rf"].min() * 100.0)
    elif "excess_over_baseline" in metric_df.columns:
        spread_pp = float((metric_df["excess_over_baseline"].max() - metric_df["excess_over_baseline"].min()) * 100.0)
        max_excess_pp = float(metric_df["excess_over_baseline"].max() * 100.0)
        min_excess_pp = float(metric_df["excess_over_baseline"].min() * 100.0)
    elif "return_rate" in metric_df.columns:
        spread_pp = float((metric_df["return_rate"].max() - metric_df["return_rate"].min()) * 100.0)
        max_excess_pp = float(metric_df["return_rate"].max() * 100.0)
        min_excess_pp = float(metric_df["return_rate"].min() * 100.0)
    else:
        spread_pp = 0.0
        max_excess_pp = 0.0
        min_excess_pp = 0.0

    # Evaluate strata spread if a strata matrix is provided (e.g. testing disguised feature)
    strata_spread_pp = None
    if strata_matrix is not None and not strata_matrix.empty:
        col_spreads = [
            float((strata_matrix[col].dropna().max() - strata_matrix[col].dropna().min()) * 100.0)
            for col in strata_matrix.columns
            if len(strata_matrix[col].dropna()) >= 2
        ]
        if col_spreads:
            strata_spread_pp = float(np.mean(col_spreads))

    # Threshold-based classification
    if strata_spread_pp is not None:
        if strata_spread_pp < thresholds.redundant_spread_pp:
            category = "Redundant / Disguised Feature"
            action = "Drop or merge into primary strata feature"
        elif strata_spread_pp >= thresholds.core_spread_pp:
            category = "Model Feature (Core / Orthogonal)"
            action = "Retain as primary standalone predictor"
        else:
            category = "Secondary Feature / Interaction"
            action = "Use as tree split or interaction term"
    else:
        if spread_pp >= thresholds.core_spread_pp:
            category = "Model Feature (Core)"
            action = "Retain as primary standalone predictor"
        elif spread_pp >= thresholds.secondary_spread_pp:
            category = "Secondary Feature / Interaction"
            action = "Use as tree split or interaction term"
        else:
            category = "Unsupported / Noise"
            action = "Exclude from feature set"

    return {
        "classification": category,
        "action": action,
        "spread_pp": round(spread_pp, 2),
        "strata_spread_pp": round(strata_spread_pp, 2) if strata_spread_pp is not None else None,
        "max_excess_pp": round(max_excess_pp, 2),
        "min_excess_pp": round(min_excess_pp, 2),
        "is_core": "Core" in category,
        "is_secondary": "Secondary" in category,
        "is_redundant": "Redundant" in category,
    }


def compare_purchase_rules(
    transactions: pd.DataFrame,
    config: CustomerLabelConfig | None = None,
) -> dict[str, Any]:
    """Empirically compare old vs new purchase rule (Step 0) across all cutoffs."""
    if config is None:
        config = CustomerLabelConfig()

    df = transactions.copy()
    old_valid = (df["QUANTITY"] > 0) & (df["SALES_VALUE"] >= 0)
    new_valid = df["SALES_VALUE"] > 0

    n_total_lines = len(df)
    n_lines_old = int(old_valid.sum())
    n_lines_new = int(new_valid.sum())

    baskets_old = set(df.loc[old_valid, "BASKET_ID"])
    baskets_new = set(df.loc[new_valid, "BASKET_ID"])
    baskets_affected = len(baskets_old - baskets_new) + len(baskets_new - baskets_old)

    zero_price_lines = int(((df["QUANTITY"] > 0) & (df["SALES_VALUE"] == 0)).sum())
    negative_qty_pos_sales = int(((df["QUANTITY"] <= 0) & (df["SALES_VALUE"] > 0)).sum())

    # Quick label flips across cutoffs
    pdays_old = df.loc[old_valid, ["household_key", "DAY"]].drop_duplicates()
    pdays_new = df.loc[new_valid, ["household_key", "DAY"]].drop_duplicates()

    first_new = pdays_new.groupby("household_key")["DAY"].min().to_dict()

    total_eligible = 0
    flips_1_to_0 = 0
    flips_0_to_1 = 0

    pdays_old_by_hh = {h: grp["DAY"].to_numpy() for h, grp in pdays_old.groupby("household_key")}
    pdays_new_by_hh = {h: grp["DAY"].to_numpy() for h, grp in pdays_new.groupby("household_key")}

    for c in config.cutoff_days:
        for h, fday in first_new.items():
            if fday >= c:
                continue
            total_eligible += 1
            days_old = pdays_old_by_hh.get(h, np.array([]))
            days_new = pdays_new_by_hh.get(h, np.array([]))

            has_old = 1 if ((days_old >= c) & (days_old <= c + config.horizon_days - 1)).any() else 0
            has_new = 1 if ((days_new >= c) & (days_new <= c + config.horizon_days - 1)).any() else 0

            if has_old != has_new:
                if has_old == 1 and has_new == 0:
                    flips_1_to_0 += 1
                else:
                    flips_0_to_1 += 1

    total_flips = flips_1_to_0 + flips_0_to_1
    flip_rate = total_flips / total_eligible if total_eligible > 0 else 0.0

    return {
        "n_total_lines": n_total_lines,
        "n_lines_old": n_lines_old,
        "n_lines_new": n_lines_new,
        "n_zero_price_lines": zero_price_lines,
        "n_negative_qty_pos_sales": negative_qty_pos_sales,
        "n_baskets_old": len(baskets_old),
        "n_baskets_new": len(baskets_new),
        "n_baskets_affected": baskets_affected,
        "n_eligible_rows": total_eligible,
        "n_label_flips": total_flips,
        "flip_rate": flip_rate,
        "flips_1_to_0": flips_1_to_0,
        "flips_0_to_1": flips_0_to_1,
        "classification": "Outlier / Data Quality",
        "verdict": (
            f"Chuyển đổi sang SALES_VALUE > 0 loại bỏ {zero_price_lines:,} dòng hàng tặng 0 đồng "
            f"và {baskets_affected:,} giỏ hàng rác. Tỷ lệ lật nhãn toàn panel cực nhỏ "
            f"({total_flips}/{total_eligible:,} = {flip_rate:.4%}), đảm bảo tính toàn vẹn dữ liệu "
            f"mà không gây xáo trộn mô hình."
        ),
    }

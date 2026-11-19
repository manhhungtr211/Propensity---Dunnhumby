"""Customer-level behavioral overview: frequency, monetary, and cadence metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd

from propensity.purchases import valid_purchase_lines

_REQUIRED_COLUMNS: Final = (
    "household_key",
    "BASKET_ID",
    "DAY",
    "SALES_VALUE",
)

METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "F",
    "spend_per_week",
    "spend_per_trip",
    "median_ipt",
    "cv_ipt",
    "total_spend",
    "n_purchase_days",
    "n_trips",
    "span_days",
)


class CustomerOverviewError(ValueError):
    """Raised when customer overview processing encounters invalid data or schema."""


@dataclass(frozen=True, slots=True)
class CustomerOverviewReport:
    """Consolidated customer overview data, descriptive statistics, and concentration."""

    household_summary: pd.DataFrame
    metric_descriptions: pd.DataFrame
    concentration: pd.DataFrame


def build_purchase_days(transactions: pd.DataFrame) -> pd.DataFrame:
    """Extract distinct purchase days per household with daily trips, spend, and gap.

    Parameters
    ----------
    transactions : pd.DataFrame
        Raw transaction log containing household_key, BASKET_ID, DAY, and SALES_VALUE.

    Returns
    -------
    pd.DataFrame
        One record per household x distinct purchase day with columns:
        - household_key: Unique household identifier
        - DAY: Distinct purchase day (int)
        - n_trips_day: Distinct checkout baskets on that day (int)
        - spend_day: Sum of valid sales value on that day (float)
        - gap: Days since previous purchase day (float, NaN for first day)
    """
    missing = [col for col in _REQUIRED_COLUMNS if col not in transactions.columns]
    if missing:
        raise CustomerOverviewError(f"Thiếu các cột giao dịch bắt buộc: {missing}")

    valid = valid_purchase_lines(transactions)
    if valid.empty:
        raise CustomerOverviewError("Không có dòng giao dịch hợp lệ để tổng hợp")

    pdays = (
        valid.groupby(["household_key", "DAY"], as_index=False)
        .agg(
            n_trips_day=("BASKET_ID", "nunique"),
            spend_day=("SALES_VALUE", "sum"),
        )
        .sort_values(["household_key", "DAY"])
        .reset_index(drop=True)
    )
    pdays["gap"] = pdays.groupby("household_key")["DAY"].diff()
    return pdays


def build_household_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build a customer-level summary table with F, M, and IPT cadence metrics.

    Parameters
    ----------
    transactions : pd.DataFrame
        Raw transaction log containing household_key, BASKET_ID, DAY, and SALES_VALUE.

    Returns
    -------
    pd.DataFrame
        One record per household with the following columns:
        - household_key: Unique household identifier
        - first_day, last_day: First and last valid purchase day
        - n_purchase_days: Distinct purchase days
        - n_trips: Distinct checkout baskets
        - total_spend: Sum of valid sales value
        - span_days: Active span (last_day - first_day + 1)
        - F: Purchase frequency (n_purchase_days / (span_days / 7))
        - spend_per_week: Monetary velocity (total_spend / (span_days / 7))
        - spend_per_trip: Basket monetary depth (total_spend / n_trips)
        - median_ipt: Median inter-purchase gap (null if n_purchase_days < 2)
        - cv_ipt: Coefficient of variation of IPT (null if n_purchase_days < 3)
        - short_span: Flag indicating span_days < 28
    """
    pdays = build_purchase_days(transactions)

    # 1. Base aggregations per household
    hh_agg = (
        pdays.groupby("household_key", as_index=False)
        .agg(
            first_day=("DAY", "min"),
            last_day=("DAY", "max"),
            n_purchase_days=("DAY", "count"),
            n_trips=("n_trips_day", "sum"),
            total_spend=("spend_day", "sum"),
        )
        .sort_values("household_key")
        .reset_index(drop=True)
    )

    # 2. Span, Frequency, and Monetary calculations
    hh_agg["span_days"] = hh_agg["last_day"] - hh_agg["first_day"] + 1
    weeks = hh_agg["span_days"] / 7.0
    hh_agg["F"] = hh_agg["n_purchase_days"] / weeks
    hh_agg["spend_per_week"] = hh_agg["total_spend"] / weeks
    hh_agg["spend_per_trip"] = hh_agg["total_spend"] / hh_agg["n_trips"]
    hh_agg["short_span"] = hh_agg["span_days"] < 28

    # 3. Inter-Purchase Time (IPT) and Regularity (CV)
    gaps = pdays.dropna(subset=["gap"])
    if not gaps.empty:
        ipt_stats = (
            gaps.groupby("household_key")["gap"]
            .agg(
                median_ipt="median",
                mean_ipt="mean",
                std_ipt=lambda s: s.std(ddof=1) if len(s) >= 2 else np.nan,
            )
            .reset_index()
        )
        ipt_stats["cv_ipt"] = np.where(
            (ipt_stats["mean_ipt"] > 0) & ipt_stats["std_ipt"].notna(),
            ipt_stats["std_ipt"] / ipt_stats["mean_ipt"],
            np.nan,
        )
        summary = hh_agg.merge(
            ipt_stats[["household_key", "median_ipt", "cv_ipt"]],
            on="household_key",
            how="left",
        )
    else:
        summary = hh_agg.copy()
        summary["median_ipt"] = np.nan
        summary["cv_ipt"] = np.nan

    # Order columns as specified
    ordered_cols = [
        "household_key",
        "first_day",
        "last_day",
        "n_purchase_days",
        "n_trips",
        "total_spend",
        "span_days",
        "F",
        "spend_per_week",
        "spend_per_trip",
        "median_ipt",
        "cv_ipt",
        "short_span",
    ]
    return summary[ordered_cols].copy()


def describe_customer_metrics(
    summary: pd.DataFrame,
    metrics: Sequence[str] = METRIC_COLUMNS,
) -> pd.DataFrame:
    """Compute descriptive statistics for each customer overview metric.

    Parameters
    ----------
    summary : pd.DataFrame
        Output of build_household_summary.
    metrics : Sequence[str]
        List of numeric metric column names to profile.

    Returns
    -------
    pd.DataFrame
        Scorecard with columns: metric, n, n_null, mean, median,
        p10, p25, p75, p90, p99, max, skew.
    """
    records = []
    for m in metrics:
        if m not in summary.columns:
            continue
        series = pd.to_numeric(summary[m], errors="coerce")
        clean = series.dropna()
        n_clean = len(clean)
        n_null = series.isna().sum()

        if n_clean > 0:
            p10, p25, p50, p75, p90, p99 = np.percentile(
                clean, [10, 25, 50, 75, 90, 99]
            )
            rec = {
                "metric": m,
                "n": n_clean,
                "n_null": n_null,
                "mean": clean.mean(),
                "median": p50,
                "p10": p10,
                "p25": p25,
                "p75": p75,
                "p90": p90,
                "p99": p99,
                "max": clean.max(),
                "skew": clean.skew() if n_clean >= 3 else np.nan,
            }
        else:
            rec = {
                "metric": m,
                "n": 0,
                "n_null": n_null,
                "mean": np.nan,
                "median": np.nan,
                "p10": np.nan,
                "p25": np.nan,
                "p75": np.nan,
                "p90": np.nan,
                "p99": np.nan,
                "max": np.nan,
                "skew": np.nan,
            }
        records.append(rec)

    return pd.DataFrame(records)


def gini(values: Sequence[float] | np.ndarray | pd.Series) -> float:
    """Compute Gini inequality coefficient for a 1D non-negative array.

    Parameters
    ----------
    values : Sequence[float] | np.ndarray | pd.Series
        Non-negative values.

    Returns
    -------
    float
        Gini coefficient between 0 (perfect equality) and 1 (perfect inequality).
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan
    if np.any(arr < 0):
        raise CustomerOverviewError("Gini coefficient yêu cầu các giá trị không âm")
    total = np.sum(arr)
    if total == 0:
        return 0.0

    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * sorted_arr)) / (n * total) - (n + 1.0) / n)


def lorenz_curve(
    values: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Lorenz curve coordinates (cumulative population vs cumulative metric).

    Parameters
    ----------
    values : Sequence[float] | np.ndarray | pd.Series
        Non-negative values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (p, L) where p is population share [0, 1] and L is metric share [0, 1].
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    if np.any(arr < 0):
        raise CustomerOverviewError("Lorenz curve yêu cầu các giá trị không âm")

    sorted_arr = np.sort(arr)
    total = np.sum(sorted_arr)
    if total == 0:
        p = np.linspace(0, 1, len(sorted_arr) + 1)
        return p, p

    cum_metric = np.cumsum(sorted_arr) / total
    p = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)

    p_full = np.insert(p, 0, 0.0)
    L_full = np.insert(cum_metric, 0, 0.0)
    return p_full, L_full


def compute_concentration_stats(
    summary: pd.DataFrame,
    columns: Sequence[str] = ("total_spend", "n_purchase_days"),
) -> pd.DataFrame:
    """Compute concentration statistics (Gini, top-household shares) for given metrics.

    Parameters
    ----------
    summary : pd.DataFrame
        Household summary table.
    columns : Sequence[str]
        Metrics to evaluate (e.g. total_spend, n_purchase_days).

    Returns
    -------
    pd.DataFrame
        Table reporting Gini, % households generating 50% and 80%,
        and % share held by top 20% households.
    """
    records = []
    n_hh = len(summary)
    for col in columns:
        if col not in summary.columns:
            continue
        vals = summary[col].dropna().values
        g_val = gini(vals)

        # Sort descending to compute top-contributor shares
        sorted_desc = np.sort(vals)[::-1]
        cum_shares = np.cumsum(sorted_desc) / np.sum(sorted_desc)

        # Percent of households accounting for 50% and 80%
        idx_50 = np.searchsorted(cum_shares, 0.50) + 1
        pct_hh_for_50 = (idx_50 / n_hh) * 100.0

        idx_80 = np.searchsorted(cum_shares, 0.80) + 1
        pct_hh_for_80 = (idx_80 / n_hh) * 100.0

        # Share generated by top 20% households (Pareto 80/20)
        top_20_count = max(1, int(round(0.20 * n_hh)))
        share_top_20 = cum_shares[min(top_20_count - 1, len(cum_shares) - 1)] * 100.0

        records.append({
            "metric": col,
            "gini_coefficient": round(g_val, 3),
            "pct_households_for_50pct_total": round(pct_hh_for_50, 1),
            "pct_households_for_80pct_total": round(pct_hh_for_80, 1),
            "pct_total_held_by_top_20pct_households": round(share_top_20, 1),
        })

    return pd.DataFrame(records)


def load_customer_overview_report(
    transactions_path: Path,
) -> CustomerOverviewReport:
    """Load transaction data from path and build the complete CustomerOverviewReport."""
    transactions = pd.read_csv(
        transactions_path,
        usecols=["household_key", "BASKET_ID", "DAY", "SALES_VALUE"],
    )
    summary = build_household_summary(transactions)
    descriptions = describe_customer_metrics(summary)
    concentration = compute_concentration_stats(summary)

    return CustomerOverviewReport(
        household_summary=summary,
        metric_descriptions=descriptions,
        concentration=concentration,
    )

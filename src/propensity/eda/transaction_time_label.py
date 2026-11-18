"""Leakage-safe transaction-time features for label comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "days_since_last_purchase",
    "active_days_90d",
    "active_weeks_90d",
    "median_interpurchase_gap_days_90d",
    "interpurchase_gap_std_days_90d",
)
FEATURE_TITLES: Final[dict[str, str]] = {
    "days_since_last_purchase": "Recency trước cutoff",
    "active_days_90d": "Số active days trong 90 ngày",
    "active_weeks_90d": "Số active weeks trong 90 ngày",
    "median_interpurchase_gap_days_90d": "Median interpurchase gap trong 90 ngày",
    "interpurchase_gap_std_days_90d": "Variability của interpurchase gap trong 90 ngày",
}
FOCUS_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "days_since_last_purchase",
    "active_days_90d",
    "interpurchase_gap_std_days_90d",
)


class TransactionTimeLabelError(ValueError):
    """Raised when inputs violate the transaction-time analysis contract."""


@dataclass(frozen=True, slots=True)
class TransactionTimeLabelReport:
    """Feature panel and label-level descriptive summary."""

    feature_panel: pd.DataFrame
    summary_by_label: pd.DataFrame


def load_transaction_time_label_report(
    label_path: Path,
    transaction_path: Path,
    observation_days: int = 90,
) -> TransactionTimeLabelReport:
    """Load source files and build the transaction-time report."""
    labels = pd.read_parquet(
        label_path,
        columns=["household_key", "cutoff_day", "label"],
    )
    transactions = pd.read_csv(
        transaction_path,
        usecols=["household_key", "DAY", "WEEK_NO", "QUANTITY", "SALES_VALUE"],
    )
    return build_transaction_time_label_report(
        labels,
        transactions,
        observation_days,
    )


def build_transaction_time_label_report(
    labels: pd.DataFrame,
    transactions: pd.DataFrame,
    observation_days: int = 90,
) -> TransactionTimeLabelReport:
    """Build leakage-safe temporal features and their label summaries."""
    feature_panel = build_transaction_time_features(
        labels,
        transactions,
        observation_days,
    )
    return TransactionTimeLabelReport(
        feature_panel=feature_panel,
        summary_by_label=summarize_transaction_time_by_label(feature_panel),
    )


def build_transaction_time_features(
    labels: pd.DataFrame,
    transactions: pd.DataFrame,
    observation_days: int = 90,
) -> pd.DataFrame:
    """Create temporal features using only transactions strictly before cutoff."""
    required_labels = {"household_key", "cutoff_day", "label"}
    required_transactions = {
        "household_key",
        "DAY",
        "WEEK_NO",
        "QUANTITY",
        "SALES_VALUE",
    }
    missing_labels = required_labels.difference(labels.columns)
    missing_transactions = required_transactions.difference(transactions.columns)
    if missing_labels:
        raise TransactionTimeLabelError(
            f"Label columns are missing: {sorted(missing_labels)}"
        )
    if missing_transactions:
        raise TransactionTimeLabelError(
            f"Transaction columns are missing: {sorted(missing_transactions)}"
        )
    if observation_days <= 0:
        raise TransactionTimeLabelError("observation_days must be positive.")
    if labels.duplicated(["household_key", "cutoff_day"]).any():
        raise TransactionTimeLabelError(
            "Labels must be unique by (household_key, cutoff_day)."
        )
    if not labels["label"].isin([0, 1]).all():
        raise TransactionTimeLabelError("Label must contain only 0 and 1.")

    valid_events = (
        transactions.loc[
            (transactions["QUANTITY"] > 0)
            & (transactions["SALES_VALUE"] >= 0),
            ["household_key", "DAY", "WEEK_NO"],
        ]
        .dropna()
        .drop_duplicates()
    )
    panels: list[pd.DataFrame] = []
    for cutoff_day in sorted(labels["cutoff_day"].unique()):
        observation_start_day = int(cutoff_day) - observation_days
        cutoff_labels = labels.loc[
            labels["cutoff_day"] == cutoff_day,
            ["household_key", "cutoff_day", "label"],
        ].copy()
        prior_events = valid_events.loc[valid_events["DAY"] < cutoff_day]
        recent_events = prior_events.loc[
            prior_events["DAY"] >= observation_start_day
        ]

        recency = (
            prior_events.groupby("household_key", as_index=False)["DAY"]
            .max()
            .assign(
                days_since_last_purchase=lambda frame, c=int(cutoff_day): c
                - frame["DAY"]
            )
            .drop(columns="DAY")
        )
        activity = recent_events.groupby("household_key", as_index=False).agg(
            active_days_90d=("DAY", "nunique"),
            active_weeks_90d=("WEEK_NO", "nunique"),
        )
        daily_events = recent_events[["household_key", "DAY"]].drop_duplicates()
        daily_events = daily_events.sort_values(["household_key", "DAY"])
        daily_events["gap_days"] = daily_events.groupby("household_key")[
            "DAY"
        ].diff()
        gap_summary = (
            daily_events.dropna(subset=["gap_days"])
            .groupby("household_key", as_index=False)
            .agg(
                median_interpurchase_gap_days_90d=("gap_days", "median"),
                interpurchase_gap_std_days_90d=("gap_days", "std"),
            )
        )

        cutoff_panel = (
            cutoff_labels.merge(
                recency,
                on="household_key",
                how="left",
                validate="one_to_one",
            )
            .merge(
                activity,
                on="household_key",
                how="left",
                validate="one_to_one",
            )
            .merge(
                gap_summary,
                on="household_key",
                how="left",
                validate="one_to_one",
            )
        )
        count_columns = ["active_days_90d", "active_weeks_90d"]
        cutoff_panel[count_columns] = (
            cutoff_panel[count_columns].fillna(0).astype("int64")
        )
        panels.append(cutoff_panel)

    result = pd.concat(panels, ignore_index=True)
    return result.sort_values(["cutoff_day", "household_key"], ignore_index=True)


def summarize_transaction_time_by_label(
    feature_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize each numerical temporal feature separately for label 0 and 1."""
    rows: list[dict[str, float | int | str]] = []
    for feature in FEATURE_COLUMNS:
        for label in (0, 1):
            values = feature_panel.loc[
                feature_panel["label"] == label,
                feature,
            ].dropna()
            rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "n_observations": len(values),
                    "missing_rate": 1 - len(values) / int(
                        feature_panel["label"].eq(label).sum()
                    ),
                    "p10": float(values.quantile(0.10)),
                    "q1": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "q3": float(values.quantile(0.75)),
                    "p90": float(values.quantile(0.90)),
                }
            )
    return pd.DataFrame(rows)


def build_focus_boxplot_stats(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Return quartiles, observed whiskers, and outlier rates for focused plots."""
    rows: list[dict[str, float | int | str]] = []
    for feature in FOCUS_FEATURE_COLUMNS:
        for label in (0, 1):
            values = feature_panel.loc[
                feature_panel["label"] == label,
                feature,
            ].dropna()
            p25 = float(values.quantile(0.25))
            p75 = float(values.quantile(0.75))
            iqr = p75 - p25
            lower_fence = p25 - 1.5 * iqr
            upper_fence = p75 + 1.5 * iqr
            inlier_mask = values.between(lower_fence, upper_fence)
            n_outliers = int((~inlier_mask).sum())
            rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "n_observations": len(values),
                    "p25": p25,
                    "p75": p75,
                    "lower_whisker": float(values.loc[inlier_mask].min()),
                    "upper_whisker": float(values.loc[inlier_mask].max()),
                    "n_outliers": n_outliers,
                    "outlier_rate": n_outliers / len(values),
                }
            )
    return pd.DataFrame(rows)


def build_focus_boxplot_caption(
    feature_panel: pd.DataFrame,
    focus_stats: pd.DataFrame,
) -> str:
    """Build a dynamic Markdown caption with dataset size and outlier-rate range."""
    by_feature = focus_stats.groupby("feature", as_index=False).agg(
        n_observations=("n_observations", "sum"),
        n_outliers=("n_outliers", "sum"),
    )
    by_feature["outlier_rate"] = (
        by_feature["n_outliers"] / by_feature["n_observations"]
    )
    minimum_rate = float(by_feature["outlier_rate"].min())
    maximum_rate = float(by_feature["outlier_rate"].max())
    return (
        f"Tập dữ liệu N = **{len(feature_panel)}**. "
        "Biểu đồ ẩn các điểm ngoại lai "
        f"(chiếm khoảng **{minimum_rate:.1%}–{maximum_rate:.1%}** "
        "số quan sát hợp lệ ở mỗi biến)."
    )

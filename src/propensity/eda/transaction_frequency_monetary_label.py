"""Frequency and monetary features from the 90 days before each cutoff."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "baskets_per_active_day_90d",
    "n_baskets_90d",
    "total_sales_value_90d",
    "avg_spend_per_basket_90d",
    "median_basket_value_90d",
    "max_basket_value_90d",
    "spend_per_active_day_90d",
)
FEATURE_TITLES: Final[dict[str, str]] = {
    "baskets_per_active_day_90d": "Số basket / active day",
    "n_baskets_90d": "Số basket",
    "total_sales_value_90d": "Tổng SALES_VALUE",
    "avg_spend_per_basket_90d": "Average spend / basket",
    "median_basket_value_90d": "Median basket value",
    "max_basket_value_90d": "Maximum basket value",
    "spend_per_active_day_90d": "Spend / active day",
}


class TransactionFrequencyMonetaryError(ValueError):
    """Raised when source data violates the feature-building contract."""


@dataclass(frozen=True, slots=True)
class TransactionFrequencyMonetaryReport:
    """Feature panel and descriptive summary separated by label."""

    feature_panel: pd.DataFrame
    summary_by_label: pd.DataFrame


def load_transaction_frequency_monetary_report(
    label_path: Path,
    transaction_path: Path,
    observation_days: int = 90,
) -> TransactionFrequencyMonetaryReport:
    """Load source files and build the frequency-monetary report."""
    labels = pd.read_parquet(
        label_path,
        columns=["household_key", "cutoff_day", "label"],
    )
    transactions = pd.read_csv(
        transaction_path,
        usecols=[
            "household_key",
            "BASKET_ID",
            "DAY",
            "QUANTITY",
            "SALES_VALUE",
        ],
    )
    return build_transaction_frequency_monetary_report(
        labels,
        transactions,
        observation_days,
    )


def build_transaction_frequency_monetary_report(
    labels: pd.DataFrame,
    transactions: pd.DataFrame,
    observation_days: int = 90,
) -> TransactionFrequencyMonetaryReport:
    """Build leakage-safe features and label-level summaries."""
    feature_panel = build_transaction_frequency_monetary_features(
        labels,
        transactions,
        observation_days,
    )
    return TransactionFrequencyMonetaryReport(
        feature_panel=feature_panel,
        summary_by_label=summarize_frequency_monetary_by_label(feature_panel),
    )


def build_transaction_frequency_monetary_features(
    labels: pd.DataFrame,
    transactions: pd.DataFrame,
    observation_days: int = 90,
) -> pd.DataFrame:
    """Create basket-grain features from [cutoff-90, cutoff-1]."""
    required_labels = {"household_key", "cutoff_day", "label"}
    required_transactions = {
        "household_key",
        "BASKET_ID",
        "DAY",
        "QUANTITY",
        "SALES_VALUE",
    }
    missing_labels = required_labels.difference(labels.columns)
    missing_transactions = required_transactions.difference(transactions.columns)
    if missing_labels:
        raise TransactionFrequencyMonetaryError(
            f"Label columns are missing: {sorted(missing_labels)}"
        )
    if missing_transactions:
        raise TransactionFrequencyMonetaryError(
            f"Transaction columns are missing: {sorted(missing_transactions)}"
        )
    if observation_days <= 0:
        raise TransactionFrequencyMonetaryError(
            "observation_days must be positive."
        )
    if labels.duplicated(["household_key", "cutoff_day"]).any():
        raise TransactionFrequencyMonetaryError(
            "Labels must be unique by (household_key, cutoff_day)."
        )
    if not labels["label"].isin([0, 1]).all():
        raise TransactionFrequencyMonetaryError("Label must contain only 0 and 1.")

    valid_lines = transactions.loc[
        (transactions["QUANTITY"] > 0) & (transactions["SALES_VALUE"] >= 0),
        ["household_key", "BASKET_ID", "DAY", "SALES_VALUE"],
    ].dropna()
    panels: list[pd.DataFrame] = []
    for cutoff_day in sorted(labels["cutoff_day"].unique()):
        observation_start_day = int(cutoff_day) - observation_days
        window_lines = valid_lines.loc[
            (valid_lines["DAY"] >= observation_start_day)
            & (valid_lines["DAY"] < cutoff_day)
        ]
        basket_values = (
            window_lines.groupby(
                ["household_key", "BASKET_ID", "DAY"],
                as_index=False,
            )["SALES_VALUE"]
            .sum()
            .rename(columns={"SALES_VALUE": "basket_value"})
        )
        household_features = basket_values.groupby(
            "household_key",
            as_index=False,
        ).agg(
            n_baskets_90d=("BASKET_ID", "size"),
            n_active_days_90d=("DAY", "nunique"),
            total_sales_value_90d=("basket_value", "sum"),
            avg_spend_per_basket_90d=("basket_value", "mean"),
            median_basket_value_90d=("basket_value", "median"),
            max_basket_value_90d=("basket_value", "max"),
        )
        household_features["baskets_per_active_day_90d"] = (
            household_features["n_baskets_90d"]
            / household_features["n_active_days_90d"]
        )
        household_features["spend_per_active_day_90d"] = (
            household_features["total_sales_value_90d"]
            / household_features["n_active_days_90d"]
        )

        cutoff_labels = labels.loc[
            labels["cutoff_day"] == cutoff_day,
            ["household_key", "cutoff_day", "label"],
        ]
        cutoff_panel = cutoff_labels.merge(
            household_features,
            on="household_key",
            how="left",
            validate="one_to_one",
        )
        zero_columns = [
            "n_baskets_90d",
            "n_active_days_90d",
            "total_sales_value_90d",
        ]
        cutoff_panel[zero_columns] = cutoff_panel[zero_columns].fillna(0)
        cutoff_panel[["n_baskets_90d", "n_active_days_90d"]] = cutoff_panel[
            ["n_baskets_90d", "n_active_days_90d"]
        ].astype("int64")
        panels.append(cutoff_panel)

    result = pd.concat(panels, ignore_index=True)
    return result.sort_values(["cutoff_day", "household_key"], ignore_index=True)


def summarize_frequency_monetary_by_label(
    feature_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize each feature independently for Negative and Positive rows."""
    rows: list[dict[str, float | int | str]] = []
    for feature in FEATURE_COLUMNS:
        for label in (0, 1):
            label_mask = feature_panel["label"].eq(label)
            values = feature_panel.loc[label_mask, feature].dropna()
            rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "n_observations": len(values),
                    "missing_rate": 1 - len(values) / int(label_mask.sum()),
                    "p10": float(values.quantile(0.10)),
                    "q1": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "q3": float(values.quantile(0.75)),
                    "p90": float(values.quantile(0.90)),
                }
            )
    return pd.DataFrame(rows)

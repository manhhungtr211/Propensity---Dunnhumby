"""Retail-discount features from the 90 days before each cutoff."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "ever_retail_discount_90d",
    "retail_discount_basket_share_90d",
    "total_retail_discount_90d",
    "retail_discount_to_sales_90d",
)
BOXPLOT_FEATURES: Final[tuple[str, ...]] = FEATURE_COLUMNS[1:]
FEATURE_TITLES: Final[dict[str, str]] = {
    "ever_retail_discount_90d": "Từng mua có RETAIL_DISC",
    "retail_discount_basket_share_90d": "Tỷ lệ basket có retail discount",
    "total_retail_discount_90d": "Tổng retail discount",
    "retail_discount_to_sales_90d": "Retail discount / sales",
}


class PromotionRetailDiscountError(ValueError):
    """Raised when source data violates the retail-discount contract."""


@dataclass(frozen=True, slots=True)
class PromotionRetailDiscountReport:
    """Feature panel and label-level descriptive summary."""

    feature_panel: pd.DataFrame
    summary_by_label: pd.DataFrame


def load_promotion_retail_discount_report(
    label_path: Path,
    transaction_path: Path,
    observation_days: int = 90,
) -> PromotionRetailDiscountReport:
    """Load source files and build the retail-discount report."""
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
            "RETAIL_DISC",
        ],
    )
    return build_promotion_retail_discount_report(
        labels,
        transactions,
        observation_days,
    )


def build_promotion_retail_discount_report(
    labels: pd.DataFrame,
    transactions: pd.DataFrame,
    observation_days: int = 90,
) -> PromotionRetailDiscountReport:
    """Build leakage-safe features and their label summaries."""
    feature_panel = build_promotion_retail_discount_features(
        labels,
        transactions,
        observation_days,
    )
    return PromotionRetailDiscountReport(
        feature_panel=feature_panel,
        summary_by_label=summarize_retail_discount_by_label(feature_panel),
    )


def build_promotion_retail_discount_features(
    labels: pd.DataFrame,
    transactions: pd.DataFrame,
    observation_days: int = 90,
) -> pd.DataFrame:
    """Create retail-discount features from [cutoff-90, cutoff-1]."""
    required_labels = {"household_key", "cutoff_day", "label"}
    required_transactions = {
        "household_key",
        "BASKET_ID",
        "DAY",
        "QUANTITY",
        "SALES_VALUE",
        "RETAIL_DISC",
    }
    if missing := required_labels.difference(labels.columns):
        raise PromotionRetailDiscountError(
            f"Label columns are missing: {sorted(missing)}"
        )
    if missing := required_transactions.difference(transactions.columns):
        raise PromotionRetailDiscountError(
            f"Transaction columns are missing: {sorted(missing)}"
        )
    if observation_days <= 0:
        raise PromotionRetailDiscountError("observation_days must be positive.")
    if labels.duplicated(["household_key", "cutoff_day"]).any():
        raise PromotionRetailDiscountError(
            "Labels must be unique by (household_key, cutoff_day)."
        )
    if not labels["label"].isin([0, 1]).all():
        raise PromotionRetailDiscountError("Label must contain only 0 and 1.")

    valid_lines = transactions.loc[
        (transactions["QUANTITY"] > 0) & (transactions["SALES_VALUE"] >= 0),
        [
            "household_key",
            "BASKET_ID",
            "DAY",
            "SALES_VALUE",
            "RETAIL_DISC",
        ],
    ].dropna(subset=["household_key", "BASKET_ID", "DAY", "SALES_VALUE"])
    valid_lines = valid_lines.copy()
    valid_lines["retail_discount_amount"] = (
        -valid_lines["RETAIL_DISC"].fillna(0)
    ).clip(lower=0)

    panels: list[pd.DataFrame] = []
    for cutoff_day in sorted(labels["cutoff_day"].unique()):
        observation_start_day = int(cutoff_day) - observation_days
        window_lines = valid_lines.loc[
            (valid_lines["DAY"] >= observation_start_day)
            & (valid_lines["DAY"] < cutoff_day)
        ]
        basket_features = window_lines.groupby(
            ["household_key", "BASKET_ID", "DAY"],
            as_index=False,
        ).agg(
            basket_sales_value=("SALES_VALUE", "sum"),
            basket_retail_discount=("retail_discount_amount", "sum"),
        )
        basket_features["has_retail_discount"] = (
            basket_features["basket_retail_discount"] > 0
        )
        household_features = basket_features.groupby(
            "household_key",
            as_index=False,
        ).agg(
            n_baskets_90d=("BASKET_ID", "size"),
            n_retail_discount_baskets_90d=("has_retail_discount", "sum"),
            total_retail_discount_90d=("basket_retail_discount", "sum"),
            total_sales_value_90d=("basket_sales_value", "sum"),
        )
        household_features["ever_retail_discount_90d"] = (
            household_features["n_retail_discount_baskets_90d"] > 0
        ).astype("int8")
        household_features["retail_discount_basket_share_90d"] = (
            household_features["n_retail_discount_baskets_90d"]
            / household_features["n_baskets_90d"]
        )
        sales_denominator = household_features["total_sales_value_90d"].where(
            household_features["total_sales_value_90d"] > 0
        )
        household_features["retail_discount_to_sales_90d"] = (
            household_features["total_retail_discount_90d"] / sales_denominator
        ).fillna(0)

        cutoff_panel = labels.loc[
            labels["cutoff_day"] == cutoff_day,
            ["household_key", "cutoff_day", "label"],
        ].merge(
            household_features[["household_key", *FEATURE_COLUMNS]],
            on="household_key",
            how="left",
            validate="one_to_one",
        )
        cutoff_panel[list(FEATURE_COLUMNS)] = cutoff_panel[
            list(FEATURE_COLUMNS)
        ].fillna(0)
        cutoff_panel["ever_retail_discount_90d"] = cutoff_panel[
            "ever_retail_discount_90d"
        ].astype("int8")
        panels.append(cutoff_panel)

    result = pd.concat(panels, ignore_index=True)
    return result.sort_values(["cutoff_day", "household_key"], ignore_index=True)


def summarize_retail_discount_by_label(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize retail-discount features independently for each label."""
    rows: list[dict[str, float | int | str]] = []
    for feature in FEATURE_COLUMNS:
        for label in (0, 1):
            values = feature_panel.loc[feature_panel["label"] == label, feature]
            rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "n_observations": len(values),
                    "mean": float(values.mean()),
                    "q1": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "q3": float(values.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)

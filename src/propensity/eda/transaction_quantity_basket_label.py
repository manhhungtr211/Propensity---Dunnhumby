"""Quantity and basket-diversity features from the pre-cutoff window."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "total_quantity_90d",
    "avg_items_per_basket_90d",
    "avg_unique_products_per_basket_90d",
    "unique_products_90d",
)
FEATURE_TITLES: Final[dict[str, str]] = {
    "total_quantity_90d": "Tổng quantity trong 90 ngày",
    "avg_items_per_basket_90d": "Tổng quantity / số basket",
    "avg_unique_products_per_basket_90d": "Unique products / basket",
    "unique_products_90d": "Unique products trong lịch sử 90 ngày",
}
FEATURE_Y_LABELS: Final[dict[str, str]] = {
    "total_quantity_90d": "Tổng QUANTITY trong 90 ngày",
    "avg_items_per_basket_90d": "Tổng QUANTITY / số basket",
    "avg_unique_products_per_basket_90d": (
        "Trung bình số PRODUCT_ID unique / basket"
    ),
    "unique_products_90d": "Số PRODUCT_ID unique trong 90 ngày",
}


class TransactionQuantityBasketError(ValueError):
    """Raised when inputs violate the quantity-basket feature contract."""


@dataclass(frozen=True, slots=True)
class TransactionQuantityBasketReport:
    """Quantity-basket feature panel and label-level summary."""

    feature_panel: pd.DataFrame
    summary_by_label: pd.DataFrame


def load_transaction_quantity_basket_report(
    label_path: Path,
    transaction_path: Path,
    observation_days: int = 90,
) -> TransactionQuantityBasketReport:
    """Load labels and transactions, then build the report."""
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
            "PRODUCT_ID",
            "QUANTITY",
            "SALES_VALUE",
        ],
    )
    return build_transaction_quantity_basket_report(
        labels,
        transactions,
        observation_days,
    )


def build_transaction_quantity_basket_report(
    labels: pd.DataFrame,
    transactions: pd.DataFrame,
    observation_days: int = 90,
) -> TransactionQuantityBasketReport:
    """Build leakage-safe features and their label summaries."""
    feature_panel = build_transaction_quantity_basket_features(
        labels,
        transactions,
        observation_days,
    )
    return TransactionQuantityBasketReport(
        feature_panel=feature_panel,
        summary_by_label=summarize_quantity_basket_by_label(feature_panel),
    )


def build_transaction_quantity_basket_features(
    labels: pd.DataFrame,
    transactions: pd.DataFrame,
    observation_days: int = 90,
) -> pd.DataFrame:
    """Create quantity and product-diversity features from cutoff-90 to cutoff-1."""
    required_labels = {"household_key", "cutoff_day", "label"}
    required_transactions = {
        "household_key",
        "BASKET_ID",
        "DAY",
        "PRODUCT_ID",
        "QUANTITY",
        "SALES_VALUE",
    }
    missing_labels = required_labels.difference(labels.columns)
    missing_transactions = required_transactions.difference(transactions.columns)
    if missing_labels:
        raise TransactionQuantityBasketError(
            f"Label columns are missing: {sorted(missing_labels)}"
        )
    if missing_transactions:
        raise TransactionQuantityBasketError(
            f"Transaction columns are missing: {sorted(missing_transactions)}"
        )
    if observation_days <= 0:
        raise TransactionQuantityBasketError("observation_days must be positive.")
    if labels.duplicated(["household_key", "cutoff_day"]).any():
        raise TransactionQuantityBasketError(
            "Labels must be unique by (household_key, cutoff_day)."
        )
    if not labels["label"].isin([0, 1]).all():
        raise TransactionQuantityBasketError("Label must contain only 0 and 1.")

    valid_lines = transactions.loc[
        (transactions["QUANTITY"] > 0) & (transactions["SALES_VALUE"] >= 0),
        ["household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY"],
    ].dropna()
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
            basket_quantity=("QUANTITY", "sum"),
            basket_unique_products=("PRODUCT_ID", "nunique"),
        )
        household_features = basket_features.groupby(
            "household_key",
            as_index=False,
        ).agg(
            total_quantity_90d=("basket_quantity", "sum"),
            avg_items_per_basket_90d=("basket_quantity", "mean"),
            avg_unique_products_per_basket_90d=(
                "basket_unique_products",
                "mean",
            ),
        )
        product_history = window_lines.groupby(
            "household_key",
            as_index=False,
        ).agg(unique_products_90d=("PRODUCT_ID", "nunique"))
        household_features = household_features.merge(
            product_history,
            on="household_key",
            how="inner",
            validate="one_to_one",
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
        zero_columns = ["total_quantity_90d", "unique_products_90d"]
        cutoff_panel[zero_columns] = cutoff_panel[zero_columns].fillna(0)
        cutoff_panel[zero_columns] = cutoff_panel[zero_columns].astype("int64")
        panels.append(cutoff_panel)

    result = pd.concat(panels, ignore_index=True)
    return result.sort_values(["cutoff_day", "household_key"], ignore_index=True)


def summarize_quantity_basket_by_label(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize each quantity-basket feature for label 0 and label 1."""
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

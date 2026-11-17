"""Brand and manufacturer history features from the pre-cutoff window."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "n_brands_90d",
    "dominant_brand_share_90d",
)
MISSING_DOMINANT_BRAND: Final = "NO_ACTIVITY_OR_PRODUCT_MAPPING"


class ProductBrandManufacturerError(ValueError):
    """Raised when inputs violate the brand-manufacturer analysis contract."""


@dataclass(frozen=True, slots=True)
class ProductBrandManufacturerReport:
    """Brand and manufacturer feature panel with label distributions."""

    feature_panel: pd.DataFrame
    numeric_summary_by_label: pd.DataFrame
    historical_brand_summary: pd.DataFrame
    dominant_brand_summary: pd.DataFrame
    manufacturer_summary: pd.DataFrame


def load_product_brand_manufacturer_report(
    label_path: Path,
    transaction_path: Path,
    product_path: Path,
    observation_days: int = 90,
) -> ProductBrandManufacturerReport:
    """Load source tables and build the report."""
    labels = pd.read_parquet(
        label_path,
        columns=["household_key", "cutoff_day", "label"],
    )
    transactions = pd.read_csv(
        transaction_path,
        usecols=[
            "household_key",
            "DAY",
            "PRODUCT_ID",
            "QUANTITY",
            "SALES_VALUE",
        ],
    )
    products = pd.read_csv(
        product_path,
        usecols=["PRODUCT_ID", "BRAND", "MANUFACTURER"],
    )
    return build_product_brand_manufacturer_report(
        labels,
        transactions,
        products,
        observation_days,
    )


def build_product_brand_manufacturer_report(
    labels: pd.DataFrame,
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    observation_days: int = 90,
) -> ProductBrandManufacturerReport:
    """Build leakage-safe brand and manufacturer history features."""
    _validate_inputs(labels, transactions, products, observation_days)
    valid_lines = transactions.loc[
        (transactions["QUANTITY"] > 0) & (transactions["SALES_VALUE"] >= 0),
        ["household_key", "DAY", "PRODUCT_ID"],
    ].dropna()
    enriched = valid_lines.merge(
        products[["PRODUCT_ID", "BRAND", "MANUFACTURER"]],
        on="PRODUCT_ID",
        how="left",
        validate="many_to_one",
    )
    panel_parts: list[pd.DataFrame] = []
    history_parts: list[pd.DataFrame] = []
    for cutoff_day in sorted(labels["cutoff_day"].unique()):
        window = enriched.loc[
            (enriched["DAY"] >= int(cutoff_day) - observation_days)
            & (enriched["DAY"] < cutoff_day)
        ]
        cutoff_labels = labels.loc[
            labels["cutoff_day"] == cutoff_day,
            ["household_key", "cutoff_day", "label"],
        ].copy()
        for dimension, column in (
            ("brand", "BRAND"),
            ("manufacturer", "MANUFACTURER"),
        ):
            dimension_lines = window.loc[
                window[column].notna() & window[column].astype(str).ne(""),
                ["household_key", column],
            ].copy()
            dimension_lines[column] = dimension_lines[column].astype(str)
            presence = dimension_lines.drop_duplicates().merge(
                cutoff_labels,
                on="household_key",
                how="inner",
                validate="many_to_one",
            )
            presence = presence.rename(columns={column: "category"})
            presence.insert(0, "dimension", dimension)
            history_parts.append(presence)

        brand_lines = window.loc[
            window["BRAND"].notna() & window["BRAND"].astype(str).ne(""),
            ["household_key", "BRAND"],
        ].copy()
        brand_lines["BRAND"] = brand_lines["BRAND"].astype(str)
        brand_counts = (
            brand_lines.groupby(["household_key", "BRAND"], as_index=False)
            .size()
            .rename(columns={"size": "n_product_lines"})
            .sort_values(
                ["household_key", "n_product_lines", "BRAND"],
                ascending=[True, False, True],
            )
        )
        brand_totals = brand_counts.groupby("household_key", as_index=False).agg(
            n_brands_90d=("BRAND", "nunique"),
            total_brand_lines_90d=("n_product_lines", "sum"),
        )
        dominant_brand = brand_counts.drop_duplicates("household_key")[
            ["household_key", "BRAND", "n_product_lines"]
        ].rename(
            columns={
                "BRAND": "dominant_brand_90d",
                "n_product_lines": "dominant_brand_lines_90d",
            }
        )
        brand_features = brand_totals.merge(
            dominant_brand,
            on="household_key",
            validate="one_to_one",
        )
        brand_features["dominant_brand_share_90d"] = (
            brand_features["dominant_brand_lines_90d"]
            / brand_features["total_brand_lines_90d"]
        )
        cutoff_panel = cutoff_labels.merge(
            brand_features,
            on="household_key",
            how="left",
            validate="one_to_one",
        )
        cutoff_panel["n_brands_90d"] = (
            cutoff_panel["n_brands_90d"].fillna(0).astype("int64")
        )
        cutoff_panel["dominant_brand_90d"] = cutoff_panel[
            "dominant_brand_90d"
        ].fillna(MISSING_DOMINANT_BRAND)
        panel_parts.append(cutoff_panel)

    feature_panel = pd.concat(panel_parts, ignore_index=True).sort_values(
        ["cutoff_day", "household_key"],
        ignore_index=True,
    )
    history = pd.concat(history_parts, ignore_index=True)
    historical = _summarize_history(history, labels)
    return ProductBrandManufacturerReport(
        feature_panel=feature_panel,
        numeric_summary_by_label=_summarize_numeric(feature_panel),
        historical_brand_summary=historical.loc[
            historical["dimension"] == "brand"
        ].reset_index(drop=True),
        dominant_brand_summary=_summarize_dominant_brand(feature_panel),
        manufacturer_summary=historical.loc[
            historical["dimension"] == "manufacturer"
        ].reset_index(drop=True),
    )


def _validate_inputs(
    labels: pd.DataFrame,
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    observation_days: int,
) -> None:
    required_labels = {"household_key", "cutoff_day", "label"}
    required_transactions = {
        "household_key",
        "DAY",
        "PRODUCT_ID",
        "QUANTITY",
        "SALES_VALUE",
    }
    required_products = {"PRODUCT_ID", "BRAND", "MANUFACTURER"}
    if missing := required_labels.difference(labels.columns):
        raise ProductBrandManufacturerError(
            f"Label columns are missing: {sorted(missing)}"
        )
    if missing := required_transactions.difference(transactions.columns):
        raise ProductBrandManufacturerError(
            f"Transaction columns are missing: {sorted(missing)}"
        )
    if missing := required_products.difference(products.columns):
        raise ProductBrandManufacturerError(
            f"Product columns are missing: {sorted(missing)}"
        )
    if observation_days <= 0:
        raise ProductBrandManufacturerError("observation_days must be positive.")
    if labels.duplicated(["household_key", "cutoff_day"]).any():
        raise ProductBrandManufacturerError(
            "Labels must be unique by (household_key, cutoff_day)."
        )
    if not products["PRODUCT_ID"].is_unique:
        raise ProductBrandManufacturerError("Products must be unique by PRODUCT_ID.")


def _summarize_history(
    history: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    label_totals = (
        labels.groupby("label", as_index=False)
        .size()
        .rename(columns={"size": "label_observations"})
    )
    summary = history.groupby(
        ["dimension", "category", "label"],
        as_index=False,
    ).agg(
        n_observations=("label", "size"),
        n_households=("household_key", "nunique"),
    )
    summary = summary.merge(label_totals, on="label", validate="many_to_one")
    summary["within_label_share"] = (
        summary["n_observations"] / summary["label_observations"]
    )
    return _add_category_metrics(summary)


def _summarize_dominant_brand(feature_panel: pd.DataFrame) -> pd.DataFrame:
    known = feature_panel.loc[
        feature_panel["dominant_brand_90d"] != MISSING_DOMINANT_BRAND
    ]
    label_totals = (
        known.groupby("label", as_index=False)
        .size()
        .rename(columns={"size": "label_observations"})
    )
    summary = (
        known.groupby(["dominant_brand_90d", "label"], as_index=False)
        .agg(
            n_observations=("label", "size"),
            n_households=("household_key", "nunique"),
        )
        .rename(columns={"dominant_brand_90d": "category"})
        .merge(label_totals, on="label", validate="many_to_one")
    )
    summary.insert(0, "dimension", "dominant_brand")
    summary["within_label_share"] = (
        summary["n_observations"] / summary["label_observations"]
    )
    return _add_category_metrics(summary)


def _add_category_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    shares = summary.pivot_table(
        index=["dimension", "category"],
        columns="label",
        values="within_label_share",
        fill_value=0,
    ).reset_index()
    shares["positive_minus_negative_pp"] = 100 * (shares[1] - shares[0])
    shares["absolute_gap_pp"] = shares["positive_minus_negative_pp"].abs()
    support = (
        summary.groupby(["dimension", "category"], as_index=False)["n_observations"]
        .sum()
        .rename(columns={"n_observations": "total_observations"})
    )
    metrics = shares.merge(
        support,
        on=["dimension", "category"],
        validate="one_to_one",
    )
    return summary.merge(
        metrics,
        on=["dimension", "category"],
        validate="many_to_one",
    )


def _summarize_numeric(feature_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for feature in NUMERIC_FEATURES:
        for label in (0, 1):
            label_mask = feature_panel["label"].eq(label)
            values = feature_panel.loc[label_mask, feature].dropna()
            rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "n_observations": len(values),
                    "missing_rate": 1 - len(values) / int(label_mask.sum()),
                    "q1": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "q3": float(values.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)

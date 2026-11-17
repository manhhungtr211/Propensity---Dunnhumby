"""Product-category history features from the 90 days before each cutoff."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

DIMENSION_COLUMNS: Final[dict[str, str]] = {
    "department": "DEPARTMENT",
    "commodity": "COMMODITY_DESC",
    "subcommodity": "SUB_COMMODITY_DESC",
}
MISSING_DOMINANT: Final = "NO_ACTIVITY_OR_PRODUCT_MAPPING"
BREADTH_FEATURES: Final[tuple[str, ...]] = (
    "n_departments_90d",
    "n_commodities_90d",
)
BREADTH_COLUMN_BY_DIMENSION: Final[dict[str, str]] = {
    "department": "n_departments_90d",
    "commodity": "n_commodities_90d",
}


class ProductCategoryLabelError(ValueError):
    """Raised when source data violates the category-analysis contract."""


@dataclass(frozen=True, slots=True)
class ProductCategoryLabelReport:
    """Category history, breadth, and dominant-category results."""

    feature_panel: pd.DataFrame
    breadth_summary_by_label: pd.DataFrame
    historical_category_summary: pd.DataFrame
    dominant_category_summary: pd.DataFrame
    overall_positive_rate: float


def load_product_category_label_report(
    label_path: Path,
    transaction_path: Path,
    product_path: Path,
    observation_days: int = 90,
) -> ProductCategoryLabelReport:
    """Load source tables and build the product-category report."""
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
        usecols=["PRODUCT_ID", *DIMENSION_COLUMNS.values()],
    )
    return build_product_category_label_report(
        labels,
        transactions,
        products,
        observation_days,
    )


def build_product_category_label_report(
    labels: pd.DataFrame,
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    observation_days: int = 90,
) -> ProductCategoryLabelReport:
    """Build leakage-safe product-category features and summaries."""
    required_labels = {"household_key", "cutoff_day", "label"}
    required_transactions = {
        "household_key",
        "DAY",
        "PRODUCT_ID",
        "QUANTITY",
        "SALES_VALUE",
    }
    required_products = {"PRODUCT_ID", *DIMENSION_COLUMNS.values()}
    if missing := required_labels.difference(labels.columns):
        raise ProductCategoryLabelError(f"Label columns are missing: {sorted(missing)}")
    if missing := required_transactions.difference(transactions.columns):
        raise ProductCategoryLabelError(
            f"Transaction columns are missing: {sorted(missing)}"
        )
    if missing := required_products.difference(products.columns):
        raise ProductCategoryLabelError(
            f"Product columns are missing: {sorted(missing)}"
        )
    if observation_days <= 0:
        raise ProductCategoryLabelError("observation_days must be positive.")
    if labels.duplicated(["household_key", "cutoff_day"]).any():
        raise ProductCategoryLabelError(
            "Labels must be unique by (household_key, cutoff_day)."
        )
    if not products["PRODUCT_ID"].is_unique:
        raise ProductCategoryLabelError("Products must be unique by PRODUCT_ID.")

    valid_lines = transactions.loc[
        (transactions["QUANTITY"] > 0) & (transactions["SALES_VALUE"] >= 0),
        ["household_key", "DAY", "PRODUCT_ID"],
    ].dropna()
    enriched_lines = valid_lines.merge(
        products[["PRODUCT_ID", *DIMENSION_COLUMNS.values()]],
        on="PRODUCT_ID",
        how="left",
        validate="many_to_one",
    )
    panel_parts: list[pd.DataFrame] = []
    history_parts: list[pd.DataFrame] = []
    for cutoff_day in sorted(labels["cutoff_day"].unique()):
        observation_start_day = int(cutoff_day) - observation_days
        window_lines = enriched_lines.loc[
            (enriched_lines["DAY"] >= observation_start_day)
            & (enriched_lines["DAY"] < cutoff_day)
        ]
        cutoff_panel = labels.loc[
            labels["cutoff_day"] == cutoff_day,
            ["household_key", "cutoff_day", "label"],
        ].copy()

        for dimension, column in DIMENSION_COLUMNS.items():
            dimension_lines = window_lines.loc[
                window_lines[column].notna() & window_lines[column].astype(str).ne(""),
                ["household_key", column],
            ]
            presence = dimension_lines.drop_duplicates().merge(
                cutoff_panel,
                on="household_key",
                how="inner",
                validate="many_to_one",
            )
            presence = presence.rename(columns={column: "category"})
            presence.insert(0, "dimension", dimension)
            history_parts.append(presence)

            if dimension == "subcommodity":
                continue
            category_counts = (
                dimension_lines.groupby(["household_key", column], as_index=False)
                .size()
                .rename(columns={"size": "n_product_lines"})
            )
            category_counts = category_counts.sort_values(
                ["household_key", "n_product_lines", column],
                ascending=[True, False, True],
            )
            dominant = (
                category_counts.drop_duplicates("household_key")
                [["household_key", column]]
                .rename(columns={column: f"dominant_{dimension}_90d"})
            )
            breadth = (
                category_counts.groupby("household_key", as_index=False)[column]
                .nunique()
                .rename(columns={column: BREADTH_COLUMN_BY_DIMENSION[dimension]})
            )
            cutoff_panel = (
                cutoff_panel.merge(
                    breadth,
                    on="household_key",
                    how="left",
                    validate="one_to_one",
                )
                .merge(
                    dominant,
                    on="household_key",
                    how="left",
                    validate="one_to_one",
                )
            )

        count_columns = list(BREADTH_FEATURES)
        cutoff_panel[count_columns] = (
            cutoff_panel[count_columns].fillna(0).astype("int64")
        )
        dominant_columns = ["dominant_department_90d", "dominant_commodity_90d"]
        cutoff_panel[dominant_columns] = cutoff_panel[dominant_columns].fillna(
            MISSING_DOMINANT
        )
        panel_parts.append(cutoff_panel)

    feature_panel = pd.concat(panel_parts, ignore_index=True).sort_values(
        ["cutoff_day", "household_key"],
        ignore_index=True,
    )
    history = pd.concat(history_parts, ignore_index=True)
    overall_rate = float(labels["label"].mean())
    return ProductCategoryLabelReport(
        feature_panel=feature_panel,
        breadth_summary_by_label=_summarize_breadth(feature_panel),
        historical_category_summary=_summarize_historical_distributions(
            history,
            labels,
        ),
        dominant_category_summary=_summarize_dominant_distributions(feature_panel),
        overall_positive_rate=overall_rate,
    )


def _summarize_breadth(feature_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for feature in BREADTH_FEATURES:
        for label in (0, 1):
            values = feature_panel.loc[feature_panel["label"] == label, feature]
            rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "n_observations": len(values),
                    "q1": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "q3": float(values.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def _summarize_historical_distributions(
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
    summary = summary.merge(
        label_totals,
        on="label",
        how="left",
        validate="many_to_one",
    )
    summary["within_label_share"] = (
        summary["n_observations"] / summary["label_observations"]
    )
    return _add_distribution_gaps(summary)


def _summarize_dominant_distributions(
    feature_panel: pd.DataFrame,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for dimension in ("department", "commodity"):
        column = f"dominant_{dimension}_90d"
        known = feature_panel.loc[feature_panel[column] != MISSING_DOMINANT]
        label_totals = (
            known.groupby("label", as_index=False)
            .size()
            .rename(columns={"size": "label_observations"})
        )
        summary = (
            known.groupby([column, "label"], as_index=False)
            .agg(
                n_observations=("label", "size"),
                n_households=("household_key", "nunique"),
            )
            .rename(columns={column: "category"})
            .merge(
                label_totals,
                on="label",
                how="left",
                validate="many_to_one",
            )
        )
        summary["within_label_share"] = (
            summary["n_observations"] / summary["label_observations"]
        )
        summary.insert(0, "dimension", dimension)
        parts.append(summary)
    return _add_distribution_gaps(pd.concat(parts, ignore_index=True))


def _add_distribution_gaps(summary: pd.DataFrame) -> pd.DataFrame:
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
    category_metrics = shares.merge(
        support,
        on=["dimension", "category"],
        validate="one_to_one",
    )
    return summary.merge(
        category_metrics,
        on=["dimension", "category"],
        how="left",
        validate="many_to_one",
    )

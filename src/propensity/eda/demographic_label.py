"""Demographic distributions conditioned on binary purchase labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from propensity.eda.demographic_label_config import (
    FEATURES,
    LABEL_NAMES,
    MISSING_GROUP,
)


class DemographicLabelError(ValueError):
    """Raised when labels or demographic inputs violate the analysis contract."""


@dataclass(frozen=True, slots=True)
class DemographicLabelReport:
    """Tables used by the demographic-label analysis notebook."""

    label_overview: pd.DataFrame
    household_label_overlap: pd.DataFrame
    overall_positive_rate: float
    positive_rate_summary: pd.DataFrame
    distribution_summary: pd.DataFrame
    category_gap_summary: pd.DataFrame
    feature_distance_summary: pd.DataFrame


def load_demographic_label_panel(
    label_path: Path,
    demographic_path: Path,
) -> pd.DataFrame:
    """Load the two sources and return one validated household-cutoff panel."""
    labels = pd.read_parquet(
        label_path,
        columns=["household_key", "cutoff_day", "label"],
    )
    demographics = pd.read_csv(
        demographic_path,
        usecols=["household_key", *FEATURES],
    )
    return prepare_demographic_label_panel(labels, demographics)


def prepare_demographic_label_panel(
    labels: pd.DataFrame,
    demographics: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and join in-memory labels and household demographics."""
    required_labels = {"household_key", "cutoff_day", "label"}
    required_demographics = {"household_key", *FEATURES}
    missing_labels = required_labels.difference(labels.columns)
    missing_demographics = required_demographics.difference(demographics.columns)
    if missing_labels:
        raise DemographicLabelError(
            f"Label columns are missing: {sorted(missing_labels)}"
        )
    if missing_demographics:
        raise DemographicLabelError(
            f"Demographic columns are missing: {sorted(missing_demographics)}"
        )
    if labels.duplicated(["household_key", "cutoff_day"]).any():
        raise DemographicLabelError(
            "Labels must be unique by (household_key, cutoff_day)."
        )
    if not demographics["household_key"].is_unique:
        raise DemographicLabelError("Demographics must be unique by household_key.")
    if not labels["label"].isin(LABEL_NAMES).all():
        raise DemographicLabelError("Label must contain only 0 and 1.")

    panel = labels[["household_key", "cutoff_day", "label"]].merge(
        demographics[["household_key", *FEATURES]],
        on="household_key",
        how="left",
        validate="many_to_one",
    )
    feature_columns = list(FEATURES)
    panel[feature_columns] = panel.loc[:, feature_columns].fillna(MISSING_GROUP)
    panel["label"] = panel["label"].astype("int8")
    panel["label_name"] = panel["label"].map(LABEL_NAMES)
    return panel


def build_demographic_label_report(panel: pd.DataFrame) -> DemographicLabelReport:
    """Build every analytical table consumed by the thin notebook."""
    distribution_summary = build_distribution_summary(panel)
    category_gap_summary, feature_distance_summary = compare_distributions(
        distribution_summary
    )
    return DemographicLabelReport(
        label_overview=build_label_overview(panel),
        household_label_overlap=build_household_label_overlap(panel),
        overall_positive_rate=float(panel["label"].mean()),
        positive_rate_summary=build_positive_rate_summary(panel),
        distribution_summary=distribution_summary,
        category_gap_summary=category_gap_summary,
        feature_distance_summary=feature_distance_summary,
    )


def build_label_overview(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize rows, households, and cutoffs in each label."""
    result = (
        panel.groupby(["label", "label_name"], as_index=False)
        .agg(
            n_rows=("household_key", "size"),
            n_households=("household_key", "nunique"),
            n_cutoffs=("cutoff_day", "nunique"),
        )
        .sort_values("label")
    )
    return result.assign(row_share=result["n_rows"] / len(panel)).reset_index(drop=True)


def build_household_label_overlap(panel: pd.DataFrame) -> pd.DataFrame:
    """Measure how often a household appears in one versus both labels."""
    labels_per_household = panel.groupby("household_key")["label"].nunique()
    n_households = len(labels_per_household)
    n_both = int(labels_per_household.eq(2).sum())
    return pd.DataFrame(
        {
            "n_households": [n_households],
            "n_households_in_both_labels": [n_both],
            "share_households_in_both_labels": [n_both / n_households],
        }
    )


def build_distribution_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Calculate P(category | label) for every demographic feature."""
    frames: list[pd.DataFrame] = []
    for feature in FEATURES:
        summary = (
            panel.groupby(["label", "label_name", feature], as_index=False)
            .agg(
                n_rows=("household_key", "size"),
                n_households=("household_key", "nunique"),
            )
            .rename(columns={feature: "category"})
        )
        summary["within_label_share"] = summary["n_rows"] / summary.groupby(
            "label"
        )["n_rows"].transform("sum")
        summary.insert(0, "feature", feature)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def build_positive_rate_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Calculate the share of label=1 rows within every demographic category."""
    frames: list[pd.DataFrame] = []
    for feature in FEATURES:
        summary = (
            panel.groupby(feature, as_index=False)
            .agg(
                n_observations=("label", "size"),
                n_positive=("label", "sum"),
                n_households=("household_key", "nunique"),
            )
            .rename(columns={feature: "category"})
        )
        summary["n_negative"] = summary["n_observations"] - summary["n_positive"]
        summary["positive_rate"] = (
            summary["n_positive"] / summary["n_observations"]
        )
        summary["is_missing_category"] = summary["category"].eq(MISSING_GROUP)
        summary.insert(0, "feature", feature)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def compare_distributions(
    distribution_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare Positive and Negative category shares without target-rate inversion."""
    known = distribution_summary.loc[
        distribution_summary["category"] != MISSING_GROUP
    ].copy()
    known["within_known_label_share"] = known["n_rows"] / known.groupby(
        ["feature", "label"]
    )["n_rows"].transform("sum")
    comparison = known.pivot_table(
        index=["feature", "category"],
        columns="label_name",
        values="within_known_label_share",
        fill_value=0,
    ).reset_index()
    comparison.columns.name = None
    comparison["positive_minus_negative_pp"] = 100 * (
        comparison["Positive"] - comparison["Negative"]
    )
    comparison["absolute_gap_pp"] = comparison[
        "positive_minus_negative_pp"
    ].abs()
    distance = (
        comparison.groupby("feature", as_index=False)["absolute_gap_pp"]
        .sum()
        .rename(columns={"absolute_gap_pp": "total_absolute_gap_pp"})
    )
    distance["total_variation_distance"] = (
        distance["total_absolute_gap_pp"] / 200
    )
    return (
        comparison.sort_values(
            ["feature", "absolute_gap_pp"], ascending=[True, False]
        ).reset_index(drop=True),
        distance.sort_values("total_variation_distance", ascending=False).reset_index(
            drop=True
        ),
    )

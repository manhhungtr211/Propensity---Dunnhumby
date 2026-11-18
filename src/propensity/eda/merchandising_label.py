"""Display and mailer exposure features from 90 days before cutoff."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

EXPOSURE_FEATURES: Final[tuple[str, ...]] = (
    "ever_display_exposure_90d",
    "ever_mailer_exposure_90d",
)
FEATURE_TITLES: Final[dict[str, str]] = {
    "ever_display_exposure_90d": "Display exposure",
    "ever_mailer_exposure_90d": "Mailer exposure",
}
CAUSAL_KEYS: Final[list[str]] = ["PRODUCT_ID", "STORE_ID", "WEEK_NO"]


class MerchandisingLabelError(ValueError):
    """Raised when merchandising inputs violate the feature contract."""


@dataclass(frozen=True, slots=True)
class MerchandisingSources:
    """Source tables required to build merchandising exposure features."""

    labels: pd.DataFrame
    transactions: pd.DataFrame
    causal: pd.DataFrame


@dataclass(frozen=True, slots=True)
class MerchandisingLabelReport:
    """Exposure feature panel and label-level summary."""

    feature_panel: pd.DataFrame
    summary_by_label: pd.DataFrame


def load_merchandising_label_report(
    label_path: Path,
    raw_dir: Path,
    observation_days: int = 90,
) -> MerchandisingLabelReport:
    """Load transactions plus relevant causal keys and build the report."""
    labels = pd.read_parquet(
        label_path,
        columns=["household_key", "cutoff_day", "label"],
    )
    transactions = pd.read_csv(
        raw_dir / "transaction_data.csv",
        usecols=[
            "household_key",
            "DAY",
            "WEEK_NO",
            "PRODUCT_ID",
            "STORE_ID",
            "QUANTITY",
            "SALES_VALUE",
        ],
    )
    global_start = int(labels["cutoff_day"].min()) - observation_days
    global_end = int(labels["cutoff_day"].max())
    relevant_lines = transactions.loc[
        (transactions["DAY"] >= global_start)
        & (transactions["DAY"] < global_end)
        & (transactions["QUANTITY"] > 0)
        & (transactions["SALES_VALUE"] >= 0)
    ]
    causal = _load_relevant_causal(
        raw_dir / "causal_data.csv",
        relevant_lines[CAUSAL_KEYS].dropna().drop_duplicates(),
    )
    return build_merchandising_label_report(
        MerchandisingSources(labels, transactions, causal),
        observation_days,
    )


def _load_relevant_causal(
    causal_path: Path,
    transaction_keys: pd.DataFrame,
) -> pd.DataFrame:
    """Read causal data in chunks and retain transaction product-store-weeks."""
    relevant_weeks = set(transaction_keys["WEEK_NO"].astype("int64"))
    keys = transaction_keys[CAUSAL_KEYS].drop_duplicates()
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        causal_path,
        usecols=[*CAUSAL_KEYS, "display", "mailer"],
        dtype={"display": "string", "mailer": "string"},
        chunksize=1_000_000,
    ):
        week_chunk = chunk.loc[chunk["WEEK_NO"].isin(relevant_weeks)]
        matched = week_chunk.merge(
            keys,
            on=CAUSAL_KEYS,
            how="inner",
        )
        if not matched.empty:
            parts.append(matched)

    if not parts:
        return pd.DataFrame(columns=[*CAUSAL_KEYS, "display", "mailer"])

    combined = pd.concat(parts, ignore_index=True)
    combined["display"] = combined["display"].astype("string").str.strip()
    combined["mailer"] = combined["mailer"].astype("string").str.strip()

    # In causal_data, a (PRODUCT_ID, STORE_ID, WEEK_NO) can appear multiple times
    # with different display/mailer codes. Aggregate to 1 row per key, prioritizing
    # active promotions (non-"0") so exposure indicators accurately reflect marketing.
    display_active = combined.loc[
        combined["display"].notna() & combined["display"].ne("0")
    ]
    best_display = display_active.drop_duplicates(
        subset=CAUSAL_KEYS, keep="first"
    )[[*CAUSAL_KEYS, "display"]]

    mailer_active = combined.loc[
        combined["mailer"].notna() & combined["mailer"].ne("0")
    ]
    best_mailer = mailer_active.drop_duplicates(
        subset=CAUSAL_KEYS, keep="first"
    )[[*CAUSAL_KEYS, "mailer"]]

    base = combined[CAUSAL_KEYS].drop_duplicates()
    base = base.merge(best_display, on=CAUSAL_KEYS, how="left")
    base = base.merge(best_mailer, on=CAUSAL_KEYS, how="left")
    base["display"] = base["display"].fillna("0")
    base["mailer"] = base["mailer"].fillna("0")

    return base.reset_index(drop=True)


def build_merchandising_label_report(
    sources: MerchandisingSources,
    observation_days: int = 90,
) -> MerchandisingLabelReport:
    """Build leakage-safe exposure features and summaries."""
    feature_panel = build_merchandising_features(sources, observation_days)
    return MerchandisingLabelReport(
        feature_panel=feature_panel,
        summary_by_label=summarize_merchandising_by_label(feature_panel),
    )


def build_merchandising_features(
    sources: MerchandisingSources,
    observation_days: int = 90,
) -> pd.DataFrame:
    """Create exposure indicators from [cutoff-90, cutoff-1]."""
    required_labels = {"household_key", "cutoff_day", "label"}
    required_transactions = {
        "household_key", "DAY", "WEEK_NO", "PRODUCT_ID", "STORE_ID",
        "QUANTITY", "SALES_VALUE",
    }
    required_causal = {*CAUSAL_KEYS, "display", "mailer"}
    for name, frame, required in (
        ("Label", sources.labels, required_labels),
        ("Transaction", sources.transactions, required_transactions),
        ("Causal", sources.causal, required_causal),
    ):
        if missing := required.difference(frame.columns):
            raise MerchandisingLabelError(
                f"{name} columns are missing: {sorted(missing)}"
            )
    if observation_days <= 0:
        raise MerchandisingLabelError("observation_days must be positive.")
    if sources.labels.duplicated(["household_key", "cutoff_day"]).any():
        raise MerchandisingLabelError(
            "Labels must be unique by (household_key, cutoff_day)."
        )

    valid_lines = sources.transactions.loc[
        (sources.transactions["QUANTITY"] > 0)
        & (sources.transactions["SALES_VALUE"] >= 0),
        ["household_key", "DAY", *CAUSAL_KEYS],
    ].dropna()
    causal = sources.causal.copy()
    causal["display"] = causal["display"].astype("string").str.strip()
    causal["mailer"] = causal["mailer"].astype("string").str.strip()
    enriched = valid_lines.merge(
        causal,
        on=CAUSAL_KEYS,
        how="left",
        validate="many_to_one",
    )
    enriched["has_causal_mapping"] = enriched[["display", "mailer"]].notna().all(axis=1)
    enriched["display_exposed"] = (
        enriched["has_causal_mapping"] & enriched["display"].ne("0")
    )
    enriched["mailer_exposed"] = (
        enriched["has_causal_mapping"] & enriched["mailer"].ne("0")
    )

    panels: list[pd.DataFrame] = []
    for cutoff_day in sorted(sources.labels["cutoff_day"].unique()):
        observation_start_day = int(cutoff_day) - observation_days
        window_lines = enriched.loc[
            (enriched["DAY"] >= observation_start_day)
            & (enriched["DAY"] < cutoff_day)
        ]
        household_features = window_lines.groupby(
            "household_key",
            as_index=False,
        ).agg(
            n_mapped_purchase_lines_90d=("has_causal_mapping", "sum"),
            ever_display_exposure_90d=("display_exposed", "max"),
            ever_mailer_exposure_90d=("mailer_exposed", "max"),
        )
        household_features[list(EXPOSURE_FEATURES)] = household_features[
            list(EXPOSURE_FEATURES)
        ].astype("boolean")
        no_mapping = household_features["n_mapped_purchase_lines_90d"].eq(0)
        household_features.loc[no_mapping, list(EXPOSURE_FEATURES)] = pd.NA
        cutoff_panel = sources.labels.loc[
            sources.labels["cutoff_day"] == cutoff_day,
            ["household_key", "cutoff_day", "label"],
        ].merge(
            household_features,
            on="household_key",
            how="left",
            validate="one_to_one",
        )
        panels.append(cutoff_panel)

    result = pd.concat(panels, ignore_index=True)
    return result.sort_values(["cutoff_day", "household_key"], ignore_index=True)


def summarize_merchandising_by_label(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize exposure rates and causal mapping coverage by label."""
    rows: list[dict[str, float | int | str]] = []
    for feature in EXPOSURE_FEATURES:
        for label in (0, 1):
            label_mask = feature_panel["label"].eq(label)
            values = feature_panel.loc[label_mask, feature].dropna().astype("bool")
            rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "n_mapped_observations": len(values),
                    "missing_mapping_rate": 1 - len(values) / int(label_mask.sum()),
                    "exposure_rate": float(values.mean()),
                }
            )
    return pd.DataFrame(rows)

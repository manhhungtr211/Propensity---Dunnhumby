"""Coupon-redemption and discount features from 90 days before cutoff."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "ever_coupon_redeemed_90d", "n_coupon_redemptions_90d",
    "days_since_last_coupon_redemption",
    "total_coupon_discount_90d", "total_coupon_match_discount_90d",
)
BOXPLOT_FEATURES: Final[tuple[str, ...]] = FEATURE_COLUMNS[1:]
FEATURE_TITLES: Final[dict[str, str]] = {
    "n_coupon_redemptions_90d": "Số coupon redeemed",
    "days_since_last_coupon_redemption": "Recency coupon redemption",
    "total_coupon_discount_90d": "Tổng COUPON_DISC",
    "total_coupon_match_discount_90d": "Tổng COUPON_MATCH_DISC",
}
FEATURE_Y_LABELS: Final[dict[str, str]] = {
    "n_coupon_redemptions_90d": "Số redemption events",
    "days_since_last_coupon_redemption": "Số ngày từ lần redeem cuối",
    "total_coupon_discount_90d": "Độ lớn coupon discount",
    "total_coupon_match_discount_90d": "Độ lớn match discount",
}


class CouponLabelError(ValueError):
    """Raised when source data violates the coupon-feature contract."""


@dataclass(frozen=True, slots=True)
class CouponSourceTables:
    """Source tables required to build coupon features."""

    labels: pd.DataFrame
    transactions: pd.DataFrame
    redemptions: pd.DataFrame


@dataclass(frozen=True, slots=True)
class CouponLabelReport:
    """Coupon feature panel and label-level descriptive summary."""

    feature_panel: pd.DataFrame
    summary_by_label: pd.DataFrame


def load_coupon_label_report(
    label_path: Path,
    raw_dir: Path,
    observation_days: int = 90,
) -> CouponLabelReport:
    """Load coupon sources and build the leakage-safe report."""
    sources = CouponSourceTables(
        labels=pd.read_parquet(
            label_path,
            columns=["household_key", "cutoff_day", "label"],
        ),
        transactions=pd.read_csv(
            raw_dir / "transaction_data.csv",
            usecols=[
                "household_key",
                "DAY",
                "QUANTITY",
                "SALES_VALUE",
                "COUPON_DISC",
                "COUPON_MATCH_DISC",
            ],
        ),
        redemptions=pd.read_csv(
            raw_dir / "coupon_redempt.csv",
            usecols=["household_key", "DAY", "COUPON_UPC", "CAMPAIGN"],
        ),
    )
    return build_coupon_label_report(sources, observation_days)


def build_coupon_label_report(
    sources: CouponSourceTables,
    observation_days: int = 90,
) -> CouponLabelReport:
    """Build coupon features and their label summaries."""
    feature_panel = build_coupon_features(sources, observation_days)
    return CouponLabelReport(
        feature_panel=feature_panel,
        summary_by_label=summarize_coupon_by_label(feature_panel),
    )


def build_coupon_features(
    sources: CouponSourceTables,
    observation_days: int = 90,
) -> pd.DataFrame:
    """Create coupon features from [cutoff-90, cutoff-1]."""
    label_columns = {"household_key", "cutoff_day", "label"}
    transaction_columns = {
        "household_key",
        "DAY",
        "QUANTITY",
        "SALES_VALUE",
        "COUPON_DISC",
        "COUPON_MATCH_DISC",
    }
    redemption_columns = {"household_key", "DAY", "COUPON_UPC", "CAMPAIGN"}
    for name, frame, required in (
        ("Label", sources.labels, label_columns),
        ("Transaction", sources.transactions, transaction_columns),
        ("Redemption", sources.redemptions, redemption_columns),
    ):
        if missing := required.difference(frame.columns):
            raise CouponLabelError(f"{name} columns are missing: {sorted(missing)}")
    if observation_days <= 0:
        raise CouponLabelError("observation_days must be positive.")
    if sources.labels.duplicated(["household_key", "cutoff_day"]).any():
        raise CouponLabelError(
            "Labels must be unique by (household_key, cutoff_day)."
        )
    if not sources.labels["label"].isin([0, 1]).all():
        raise CouponLabelError("Label must contain only 0 and 1.")

    redemptions = sources.redemptions[
        ["household_key", "DAY", "COUPON_UPC", "CAMPAIGN"]
    ].dropna().drop_duplicates()
    valid_lines = sources.transactions.loc[
        (sources.transactions["QUANTITY"] > 0)
        & (sources.transactions["SALES_VALUE"] >= 0),
        ["household_key", "DAY", "COUPON_DISC", "COUPON_MATCH_DISC"],
    ].dropna(subset=["household_key", "DAY"])
    valid_lines = valid_lines.copy()
    valid_lines["coupon_discount_amount"] = (
        -valid_lines["COUPON_DISC"].fillna(0)
    ).clip(lower=0)
    valid_lines["coupon_match_discount_amount"] = (
        -valid_lines["COUPON_MATCH_DISC"].fillna(0)
    ).clip(lower=0)

    panels: list[pd.DataFrame] = []
    for cutoff_day in sorted(sources.labels["cutoff_day"].unique()):
        observation_start_day = int(cutoff_day) - observation_days
        redemption_window = redemptions.loc[
            (redemptions["DAY"] >= observation_start_day)
            & (redemptions["DAY"] < cutoff_day)
        ]
        redemption_features = redemption_window.groupby(
            "household_key",
            as_index=False,
        ).agg(
            n_coupon_redemptions_90d=("COUPON_UPC", "size"),
            last_coupon_redemption_day=("DAY", "max"),
        )
        redemption_features["ever_coupon_redeemed_90d"] = 1
        redemption_features["days_since_last_coupon_redemption"] = (
            int(cutoff_day) - redemption_features["last_coupon_redemption_day"]
        )

        transaction_window = valid_lines.loc[
            (valid_lines["DAY"] >= observation_start_day)
            & (valid_lines["DAY"] < cutoff_day)
        ]
        discount_features = transaction_window.groupby(
            "household_key",
            as_index=False,
        ).agg(
            total_coupon_discount_90d=("coupon_discount_amount", "sum"),
            total_coupon_match_discount_90d=(
                "coupon_match_discount_amount",
                "sum",
            ),
        )
        household_features = redemption_features.merge(
            discount_features,
            on="household_key",
            how="outer",
            validate="one_to_one",
        )
        cutoff_panel = sources.labels.loc[
            sources.labels["cutoff_day"] == cutoff_day,
            ["household_key", "cutoff_day", "label"],
        ].merge(
            household_features[["household_key", *FEATURE_COLUMNS]],
            on="household_key",
            how="left",
            validate="one_to_one",
        )
        zero_columns = [
            "ever_coupon_redeemed_90d", "n_coupon_redemptions_90d",
            "total_coupon_discount_90d", "total_coupon_match_discount_90d",
        ]
        cutoff_panel[zero_columns] = cutoff_panel[zero_columns].fillna(0)
        cutoff_panel[["ever_coupon_redeemed_90d", "n_coupon_redemptions_90d"]] = (
            cutoff_panel[
                ["ever_coupon_redeemed_90d", "n_coupon_redemptions_90d"]
            ].astype("int64")
        )
        panels.append(cutoff_panel)

    result = pd.concat(panels, ignore_index=True)
    return result.sort_values(["cutoff_day", "household_key"], ignore_index=True)


def summarize_coupon_by_label(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize each coupon feature independently for both labels."""
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
                    "q1": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "q3": float(values.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)

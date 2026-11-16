#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///

# ─── How to run ───
# From the repository root:
#   uv run python scripts/build_purchase_feature_dataset.py
# ──────────────────

"""Build the all-table T=30, H=7 household-cutoff dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from propensity.eda.dunnhumby import DUNNHUMBY_TRANSACTION_PROFILE_CONFIG
from propensity.eda.dunnhumby_feature_panel import (
    ALL_TABLE_FEATURE_COLUMNS,
    DunnhumbyFeatureSources,
    build_dunnhumby_feature_panel,
)
from propensity.eda.purchase_feature_panel import (
    build_feature_target_associations,
)
from propensity.eda.purchase_target_panel import PurchaseTargetSpec

from propensity.io import read_csv
from propensity.schema.dunnhumby import TRANSACTION_SPEC

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
RAW_DIR: Final = PROJECT_ROOT / "data" / "raw" / "Dunnhumby"
OUTPUT_DIR: Final = (
    PROJECT_ROOT / "data" / "processed" / "purchase_propensity_t30_h7_step14"
)
PANEL_PATH: Final = OUTPUT_DIR / "household_cutoff_features.parquet"
ASSOCIATION_PATH: Final = OUTPUT_DIR / "feature_target_associations.parquet"


def main() -> None:
    """Materialize the feature-target panel and cutoff-level associations."""
    sources = DunnhumbyFeatureSources(
        transactions=read_csv(
            RAW_DIR / "transaction_data.csv",
            spec=TRANSACTION_SPEC,
        ),
        products=read_csv(RAW_DIR / "product.csv"),
        demographics=read_csv(RAW_DIR / "hh_demographic.csv"),
        campaign_table=read_csv(RAW_DIR / "campaign_table.csv"),
        campaign_desc=read_csv(RAW_DIR / "campaign_desc.csv"),
        coupons=read_csv(RAW_DIR / "coupon.csv"),
        coupon_redemptions=read_csv(RAW_DIR / "coupon_redempt.csv"),
        causal=read_csv(RAW_DIR / "causal_data.csv"),
    )
    spec = PurchaseTargetSpec(
        observation_days=30,
        horizon_days=7,
        first_cutoff_day=31,
        cutoff_step_days=14,
    )
    panel = build_dunnhumby_feature_panel(
        sources,
        DUNNHUMBY_TRANSACTION_PROFILE_CONFIG,
        spec,
    )
    associations = build_feature_target_associations(
        panel,
        feature_columns=ALL_TABLE_FEATURE_COLUMNS,
        min_samples=300,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH, compression="zstd", index=False)
    associations.to_parquet(ASSOCIATION_PATH, compression="zstd", index=False)
    print(f"Created {PANEL_PATH} ({panel.shape[0]:,} rows, {panel.shape[1]} columns)")
    print(
        f"Created {ASSOCIATION_PATH} "
        f"({associations.shape[0]:,} rows, {associations.shape[1]} columns)"
    )


if __name__ == "__main__":
    main()

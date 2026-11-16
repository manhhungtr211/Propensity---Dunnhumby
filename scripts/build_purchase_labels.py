"""Materialize the T90/H7/step7 household purchase-label panel."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from propensity.io import read_csv
from propensity.labels import DEFAULT_PURCHASE_LABEL_SPEC, build_purchase_labels
from propensity.schema.dunnhumby import TRANSACTION_SPEC

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
TRANSACTION_PATH: Final = (
    PROJECT_ROOT / "data" / "raw" / "Dunnhumby" / "transaction_data.csv"
)
OUTPUT_PATH: Final = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "purchase_propensity_t90_h7_step7"
    / "household_cutoff_labels.parquet"
)


def main() -> None:
    """Build and save the configured household-cutoff label panel."""
    transactions = read_csv(TRANSACTION_PATH, spec=TRANSACTION_SPEC)
    labels = build_purchase_labels(transactions, DEFAULT_PURCHASE_LABEL_SPEC)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(OUTPUT_PATH, index=False, compression="zstd")
    positive_rate = float(labels["label"].mean())
    print(
        f"Created {OUTPUT_PATH} | rows={len(labels):,} | "
        f"cutoffs={labels['cutoff_day'].nunique()} | "
        f"positive_rate={positive_rate:.2%}"
    )


if __name__ == "__main__":
    main()

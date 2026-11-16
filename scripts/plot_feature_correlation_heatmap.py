#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8", "numpy>=2.0", "pandas>=2.2", "pyarrow>=15"]
# ///

"""Join raw Dunnhumby tables to transactions and plot raw numeric correlations."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from propensity.io import read_csv
from propensity.schema.dunnhumby import TRANSACTION_SPEC

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
RAW_DIR: Final = PROJECT_ROOT / "data" / "raw" / "Dunnhumby"
DATA_DIR: Final = (
    PROJECT_ROOT / "data" / "processed" / "purchase_propensity_t30_h7_step14"
)
PANEL_PATH: Final = DATA_DIR / "household_cutoff_features.parquet"
MATRIX_PATH: Final = DATA_DIR / "feature_correlation_matrix.csv"
FIGURE_PATH: Final = DATA_DIR / "feature_correlation_heatmap.png"
MIN_SAMPLES: Final = 300

# Original numeric, non-ID columns only. Prefixes disambiguate repeated raw names.
RAW_NUMERIC_COLUMNS: Final = (
    ("transaction_data.DAY", "DAY"),
    ("transaction_data.QUANTITY", "QUANTITY"),
    ("transaction_data.SALES_VALUE", "SALES_VALUE"),
    ("transaction_data.RETAIL_DISC", "RETAIL_DISC"),
    ("transaction_data.TRANS_TIME", "TRANS_TIME"),
    ("transaction_data.WEEK_NO", "WEEK_NO"),
    ("transaction_data.COUPON_DISC", "COUPON_DISC"),
    ("transaction_data.COUPON_MATCH_DISC", "COUPON_MATCH_DISC"),
    ("campaign_desc.START_DAY", "START_DAY"),
    ("campaign_desc.END_DAY", "END_DAY"),
    ("coupon_redempt.DAY", "REDEMPTION_DAY"),
    ("target", "target"),
)


def build_joined_numeric_frame() -> pd.DataFrame:
    """Join every raw table around transaction rows without engineering features."""
    transactions = read_csv(
        RAW_DIR / "transaction_data.csv",
        spec=TRANSACTION_SPEC,
    )
    products = read_csv(RAW_DIR / "product.csv")
    demographics = read_csv(RAW_DIR / "hh_demographic.csv")
    causal = read_csv(RAW_DIR / "causal_data.csv")
    campaigns = read_csv(RAW_DIR / "campaign_table.csv").merge(
        read_csv(RAW_DIR / "campaign_desc.csv"),
        on="CAMPAIGN",
        how="left",
        suffixes=("", "_campaign_desc"),
    )
    coupons = read_csv(RAW_DIR / "coupon.csv")
    redemptions = read_csv(RAW_DIR / "coupon_redempt.csv").rename(
        columns={"DAY": "REDEMPTION_DAY"}
    )
    panel = pd.read_parquet(PANEL_PATH)
    targets = panel.loc[
        panel["has_full_observation_window"],
        ["household_key", "cutoff_day", "target"],
    ]

    joined = transactions.merge(targets, on="household_key", how="inner")
    joined = joined.loc[
        joined["DAY"].between(joined["cutoff_day"] - 29, joined["cutoff_day"])
    ]
    joined = (
        joined.merge(products, on="PRODUCT_ID", how="left")
        .merge(demographics, on="household_key", how="left")
        .merge(causal, on=["PRODUCT_ID", "STORE_ID", "WEEK_NO"], how="left")
        .merge(campaigns, on="household_key", how="left")
        .merge(coupons, on=["CAMPAIGN", "PRODUCT_ID"], how="left")
        .merge(
            redemptions,
            on=["household_key", "CAMPAIGN", "COUPON_UPC"],
            how="left",
        )
    )
    renamed = {source: label for label, source in RAW_NUMERIC_COLUMNS}
    return joined.loc[:, ["cutoff_day", *renamed]].rename(columns=renamed)


def correlation_matrix(
    joined: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Calculate each cutoff separately, then return across-cutoff medians."""
    labels = [label for label, _ in RAW_NUMERIC_COLUMNS]
    size = len(labels)
    matrix = np.full((size, size), np.nan)
    sample_sizes = np.zeros((size, size), dtype=np.int64)
    cutoff_groups = list(joined.groupby("cutoff_day", sort=True))
    for row_index in range(size):
        for column_index in range(size):
            correlations: list[float] = []
            counts: list[int] = []
            for _, group in cutoff_groups:
                pair = group[[labels[row_index], labels[column_index]]].dropna()
                if len(pair) >= MIN_SAMPLES:
                    correlations.append(float(pair.corr().iloc[0, 1]))
                    counts.append(len(pair))
            if correlations:
                matrix[row_index, column_index] = float(np.median(correlations))
                sample_sizes[row_index, column_index] = int(np.median(counts))
    return matrix, sample_sizes, len(joined)


def main() -> None:
    """Join raw tables, export the matrix, and render the heatmap."""
    labels = [label for label, _ in RAW_NUMERIC_COLUMNS]
    matrix, sample_sizes, joined_rows = correlation_matrix(build_joined_numeric_frame())
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "feature": labels,
            **{column: matrix[:, index] for index, column in enumerate(labels)},
        }
    ).to_csv(MATRIX_PATH, index=False)

    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    visible = np.ma.array(matrix, mask=mask)
    figure, axis = plt.subplots(figsize=(14, 11), constrained_layout=True)
    image = axis.imshow(visible, cmap="RdBu_r", vmin=-1, vmax=1)
    for row_index in range(len(labels)):
        for column_index in range(row_index + 1):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}\nn={sample_sizes[row_index, column_index]:,}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(value) >= 0.55 else "#172033",
                )
    axis.set_xticks(range(len(labels)), labels, rotation=48, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title(
        "Raw source columns versus T30/H7 purchase target",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )
    axis.set_xlabel(
        f"Median Pearson correlation across cutoffs; IDs removed; "
        f"minimum {MIN_SAMPLES:,} valid rows/cutoff; joined rows = {joined_rows:,}",
        labelpad=12,
    )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.8, pad=0.025)
    colorbar.set_label("Pearson correlation", rotation=270, labelpad=18)
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Created {MATRIX_PATH}")
    print(f"Created {FIGURE_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib>=3.8",
#     "polars>=1.0",
# ]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run scripts/plot_customer_return_intervals.py
# 3. Or make executable and run:
#      chmod +x scripts/plot_customer_return_intervals.py && ./scripts/plot_customer_return_intervals.py
# ──────────────────

"""Export the customer return-interval chart used by transaction EDA."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib
import matplotlib.pyplot as plt
import polars as pl

from propensity.eda.dunnhumby import DUNNHUMBY_TRANSACTION_PROFILE_CONFIG
from propensity.eda.transaction import profile_return_intervals
from propensity.eda.visualization import plot_return_interval_distribution

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
TRANSACTION_PATH: Final = (
    PROJECT_ROOT / "data" / "raw" / "Dunnhumby" / "transaction_data.csv"
)
OUTPUT_PATH: Final = (
    PROJECT_ROOT
    / "slides"
    / "transaction_feature_understanding"
    / "figures"
    / "customer_return_interval_distribution.png"
)


def main() -> None:
    """Create the customer return-interval distribution chart."""
    matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    transactions = pl.scan_csv(TRANSACTION_PATH)
    distribution = profile_return_intervals(
        transactions,
        DUNNHUMBY_TRANSACTION_PROFILE_CONFIG,
    )
    figure = plot_return_interval_distribution(distribution)
    figure.text(
        0.01,
        0.015,
        "Nguồn: transaction_data.csv | 223.033 khoảng quay lại từ 2.500 households",
        fontsize=8.5,
        color="#6B7280",
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Created {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

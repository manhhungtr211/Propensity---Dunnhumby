"""CLI entrypoint to run data overview profiling and export summary tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from propensity.overview import RESULT_CONTRACTS, build_dunnhumby_overview


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile Dunnhumby tables across structure, grain, key, time, and quality."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/Dunnhumby"),
        help="Path to raw Dunnhumby CSV files (default: data/raw/Dunnhumby)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/data_overview"),
        help="Directory to save exported summary CSVs (default: data/processed/data_overview)",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=None,
        help="Optional subset of Dunnhumby tables to profile (e.g. campaign_desc product)",
    )

    args = parser.parse_args()

    if not args.raw_dir.exists():
        print(f"Error: Raw directory does not exist: {args.raw_dir}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting data overview for raw_dir={args.raw_dir}...")
    report = build_dunnhumby_overview(raw_dir=args.raw_dir, tables=args.tables)

    for contract in RESULT_CONTRACTS:
        frame = getattr(report, contract.name)
        out_file = args.output_dir / f"{contract.name}.csv"
        frame.to_csv(out_file, index=False)
        print(f"  - Saved {contract.name} ({len(frame)} rows) -> {out_file}")

    print("\nData overview complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

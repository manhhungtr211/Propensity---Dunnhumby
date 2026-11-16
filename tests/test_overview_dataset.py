"""Behavioral tests for packaged single-dataset and multi-dataset profiling."""

from pathlib import Path

import pandas as pd

from propensity.overview import (
    OverviewReport,
    build_dunnhumby_overview,
    profile_dataset,
)
from propensity.schema.dunnhumby import TRANSACTION_SPEC


def test_profile_dataset_packages_all_sections_in_one_call() -> None:
    # Given: a dataset with transaction-like structure
    frame = pd.DataFrame(
        {
            "household_key": pd.Series([101, 102, 102], dtype="Int64"),
            "BASKET_ID": pd.Series([1, 2, 2], dtype="Int64"),
            "PRODUCT_ID": pd.Series([50, 50, 50], dtype="Int64"),
            "DAY": [10, 12, 12],
            "WEEK_NO": [2, 2, 2],
            "TRANS_TIME": [900, 1400, 1400],
            "STORE_ID": pd.Series([1, 1, 1], dtype="Int64"),
            "QUANTITY": [1, 2, 2],
            "SALES_VALUE": [2.5, 5.0, 5.0],
        }
    )

    # When: calling the consolidated package function
    report = profile_dataset(frame, spec=TRANSACTION_SPEC)

    # Then: report is an OverviewReport with all 7 contract DataFrames populated
    assert isinstance(report, OverviewReport)

    # 1. Structure
    assert len(report.table_summary) == 1
    assert report.table_summary.loc[0, "table"] == "transaction"
    assert report.table_summary.loc[0, "n_rows"] == 3
    assert report.table_summary.loc[0, "n_columns"] == 9

    # 2. Columns
    assert len(report.column_summary) == 9
    assert "household_key" in report.column_summary["column"].tolist()

    # 3. Identifiers
    assert len(report.identifier_summary) == 4
    hh_row = report.identifier_summary[
        report.identifier_summary["column"] == "household_key"
    ].iloc[0]
    assert hh_row["n_rows"] == 3
    assert hh_row["n_unique_non_missing"] == 2

    # 4. Grain
    assert len(report.grain_summary) == 1
    grain_row = report.grain_summary.iloc[0]
    assert grain_row["grain"] == "BASKET_ID, PRODUCT_ID"
    # Row 1 and 2 both have (BASKET_ID=2, PRODUCT_ID=50) -> duplicate group!
    assert grain_row["duplicate_groups"] == 1
    assert grain_row["excess_rows"] == 1
    assert bool(grain_row["is_unique"]) is False

    # 5. Time coverage
    assert len(report.time_summary) >= 3  # DAY, WEEK_NO, TRANS_TIME

    # 6. Quality
    assert len(report.quality_summary) == 10  # 9 cols + 1 table row

    # 7. Numeric distribution (df.describe equivalent)
    assert len(report.numeric_summary) >= 2  # QUANTITY, SALES_VALUE
    assert "SALES_VALUE" in report.numeric_summary["column"].tolist()
    assert "QUANTITY" in report.numeric_summary["column"].tolist()

    # 8. Provenance
    assert len(report.provenance) == 1
    assert report.provenance.loc[0, "scan_mode"] == "memory"
    assert bool(report.provenance.loc[0, "exact"]) is True


def test_build_dunnhumby_overview_on_small_tables(tmp_path: Path) -> None:
    # Given: a directory with small Dunnhumby CSV files
    campaign_file = tmp_path / "campaign_desc.csv"
    pd.DataFrame(
        {
            "DESCRIPTION": ["TypeA", "TypeB"],
            "CAMPAIGN": [1, 2],
            "START_DAY": [224, 300],
            "END_DAY": [264, 350],
        }
    ).to_csv(campaign_file, index=False)

    # When: running orchestrator on selected table
    report = build_dunnhumby_overview(raw_dir=tmp_path, tables=("campaign_desc",))

    # Then
    assert isinstance(report, OverviewReport)
    assert len(report.table_summary) == 1
    assert report.table_summary.loc[0, "table"] == "campaign_desc"
    assert report.table_summary.loc[0, "n_rows"] == 2

    grain_row = report.grain_summary.iloc[0]
    assert grain_row["table"] == "campaign_desc"
    assert grain_row["grain"] == "CAMPAIGN"
    assert bool(grain_row["is_unique"]) is True
    assert bool(grain_row["is_valid"]) is True

    interval_row = report.time_summary[report.time_summary["kind"] == "interval"].iloc[
        0
    ]
    assert interval_row["start_column"] == "START_DAY"
    assert interval_row["end_column"] == "END_DAY"
    assert interval_row["min"] == 224
    assert interval_row["max"] == 350

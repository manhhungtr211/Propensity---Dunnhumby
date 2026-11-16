"""Behavioral tests for basic quality profiling."""

import pandas as pd

from propensity.overview.models import QUALITY_SUMMARY_CONTRACT
from propensity.overview.quality import profile_quality


def test_quality_profiling_column_metrics_and_table_duplicates() -> None:
    # Given:
    # row 0 and row 1 are exact duplicates
    # col "a" has one missing (None), one blank ("  "), and values ("x", "x")
    # col "b" is constant (all 1)
    # col "c" is all missing
    frame = pd.DataFrame(
        {
            "a": ["x", "x", "  ", None],
            "b": [1, 1, 1, 1],
            "c": [None, None, " ", None],
        }
    )

    # When
    result = profile_quality(frame, table_name="test_quality")

    # Then
    assert tuple(result.columns) == QUALITY_SUMMARY_CONTRACT.columns
    # 3 column rows + 1 table row = 4 rows
    assert len(result) == 4

    col_a = result[result["column"] == "a"].iloc[0]
    assert col_a["n_rows"] == 4
    assert col_a["n_blank"] == 1
    assert col_a["n_missing"] == 2
    assert col_a["missing_pct"] == 50.0
    assert col_a["n_unique_non_missing"] == 1
    assert bool(col_a["is_all_missing"]) is False
    assert bool(col_a["is_constant"]) is True

    col_b = result[result["column"] == "b"].iloc[0]
    assert col_b["n_missing"] == 0
    assert col_b["missing_pct"] == 0.0
    assert col_b["n_unique_non_missing"] == 1
    assert bool(col_b["is_all_missing"]) is False
    assert bool(col_b["is_constant"]) is True

    col_c = result[result["column"] == "c"].iloc[0]
    assert col_c["n_missing"] == 4
    assert col_c["missing_pct"] == 100.0
    assert col_c["n_unique_non_missing"] == 0
    assert bool(col_c["is_all_missing"]) is True
    assert (
        bool(col_c["is_constant"]) is False
    )  # All-missing must NOT be marked constant!

    table_row = result[result["column"].isna()].iloc[0]
    assert table_row["n_rows"] == 4
    assert table_row["duplicate_groups"] == 1
    assert table_row["excess_rows"] == 1
    assert table_row["duplicate_pct"] == 25.0


def test_quality_profiling_empty_frame() -> None:
    # Given
    frame = pd.DataFrame(columns=["a", "b"])

    # When
    result = profile_quality(frame, table_name="empty")

    # Then
    assert len(result) == 3  # 2 columns + 1 table row
    col_a = result[result["column"] == "a"].iloc[0]
    assert pd.isna(col_a["missing_pct"])
    assert bool(col_a["is_all_missing"]) is False
    assert bool(col_a["is_constant"]) is False

    table_row = result[result["column"].isna()].iloc[0]
    assert table_row["duplicate_groups"] == 0
    assert table_row["excess_rows"] == 0
    assert pd.isna(table_row["duplicate_pct"])

"""Behavioral tests for table preview helper functions."""

from pathlib import Path

import pandas as pd

from propensity.overview.preview import preview_dunnhumby_tables, preview_table


def test_preview_table_custom_rows_and_columns() -> None:
    # Given
    frame = pd.DataFrame({
        "A": [1, 2, 3, 4, 5, 6],
        "B": ["a", "b", "c", "d", "e", "f"],
        "C": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })

    # When: customize columns and n_rows
    result_head = preview_table(frame, columns=["A", "C"], n_rows=3, mode="head")
    result_tail = preview_table(frame, columns=["B"], n_rows=2, mode="tail")
    result_sample = preview_table(frame, n_rows=2, mode="sample", random_state=42)

    # Then
    assert list(result_head.columns) == ["A", "C"]
    assert len(result_head) == 3
    assert result_head["A"].tolist() == [1, 2, 3]

    assert list(result_tail.columns) == ["B"]
    assert len(result_tail) == 2
    assert result_tail["B"].tolist() == ["e", "f"]

    assert len(result_sample) == 2
    assert list(result_sample.columns) == ["A", "B", "C"]


def test_preview_table_ignores_missing_column_names() -> None:
    # Given
    frame = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    # When: request existing column A and non-existing column Z
    result = preview_table(frame, columns=["A", "Z"], n_rows=2)

    # Then: only existing column A is retained
    assert list(result.columns) == ["A"]
    assert len(result) == 2


def test_preview_dunnhumby_tables_from_dir(tmp_path: Path) -> None:
    # Given: temporary CSV files
    (tmp_path / "campaign_desc.csv").write_text(
        "DESCRIPTION,CAMPAIGN,START_DAY,END_DAY\nTypeA,1,100,150\nTypeB,2,200,250\nTypeC,3,300,350\n"
    )

    # When: previewing tables
    previews = preview_dunnhumby_tables(
        raw_dir=tmp_path,
        tables=("campaign_desc",),
        n_rows=2,
        columns={"campaign_desc": ["CAMPAIGN", "DESCRIPTION"]},
    )

    # Then
    assert "campaign_desc" in previews
    df = previews["campaign_desc"]
    assert len(df) == 2
    assert list(df.columns) == ["CAMPAIGN", "DESCRIPTION"]
    assert df["CAMPAIGN"].tolist() == [1, 2]

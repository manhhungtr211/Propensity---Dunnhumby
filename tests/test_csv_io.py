from __future__ import annotations

import polars as pl
import pytest

from propensity.io import scan_csv


def test_scan_csv_returns_lazy_data_from_caller_path(tmp_path) -> None:
    # Given
    csv_path = tmp_path / "transactions.csv"
    csv_path.write_text("household_key,DAY\n1,10\n", encoding="utf-8")

    # When
    result = scan_csv(csv_path).collect()

    # Then
    assert result.to_dict(as_series=False) == {"household_key": [1], "DAY": [10]}


def test_scan_csv_applies_schema_overrides(tmp_path) -> None:
    # Given
    csv_path = tmp_path / "codes.csv"
    csv_path.write_text("code\n001\n", encoding="utf-8")

    # When
    result = scan_csv(csv_path, schema_overrides={"code": pl.String}).collect()

    # Then
    assert result.schema["code"] == pl.String
    assert result["code"].item() == "001"


def test_scan_csv_fails_at_boundary_when_file_is_missing(tmp_path) -> None:
    # Given
    missing_path = tmp_path / "missing.csv"

    # When / Then
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        scan_csv(missing_path)

"""Behavioral tests for table and column structure profiling."""

import pandas as pd
import pytest

from propensity.overview.models import (
    COLUMN_SUMMARY_CONTRACT,
    TABLE_SUMMARY_CONTRACT,
)
from propensity.overview.structure import profile_columns, profile_table_structure
from propensity.schema.spec import ColumnRole, DatasetSpec, DatasetSpecError


def test_profile_table_structure_returns_contract_columns() -> None:
    # Given
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    # When
    result = profile_table_structure(frame, table_name="test_table")

    # Then
    assert tuple(result.columns) == TABLE_SUMMARY_CONTRACT.columns
    assert len(result) == 1
    assert result.loc[0, "table"] == "test_table"
    assert result.loc[0, "n_rows"] == 3
    assert result.loc[0, "n_columns"] == 2
    assert result.loc[0, "memory_bytes"] > 0


def test_profile_table_structure_empty_frame() -> None:
    # Given
    frame = pd.DataFrame(columns=["a", "b"])

    # When
    result = profile_table_structure(frame, table_name="empty_table")

    # Then
    assert len(result) == 1
    assert result.loc[0, "table"] == "empty_table"
    assert result.loc[0, "n_rows"] == 0
    assert result.loc[0, "n_columns"] == 2


def test_profile_columns_preserves_order_and_semantics() -> None:
    # Given
    frame = pd.DataFrame(
        {
            "household_key": pd.Series([10, 20], dtype="Int64"),
            "DAY": [1, 2],
            "SALES": [5.5, 10.0],
            "FLAG": [True, False],
        }
    )
    spec = DatasetSpec(
        name="test_table",
        entity_col="household_key",
        time_col="DAY",
        grain_cols=("household_key", "DAY"),
        identifier_cols=("household_key",),
        ordinal_time_cols=("DAY",),
        required_cols=("SALES",),
    )

    # When
    result = profile_columns(frame, table_name="test_table", spec=spec)

    # Then
    assert tuple(result.columns) == COLUMN_SUMMARY_CONTRACT.columns
    assert len(result) == 4
    assert result["column"].tolist() == ["household_key", "DAY", "SALES", "FLAG"]

    hh_row = result[result["column"] == "household_key"].iloc[0]
    assert hh_row["role"] == ColumnRole.ENTITY
    assert bool(hh_row["is_identifier"]) is True
    assert bool(hh_row["is_grain"]) is True
    assert bool(hh_row["is_required"]) is True

    day_row = result[result["column"] == "DAY"].iloc[0]
    assert day_row["role"] == ColumnRole.TIME
    assert bool(day_row["is_time"]) is True
    assert bool(day_row["is_grain"]) is True

    sales_row = result[result["column"] == "SALES"].iloc[0]
    assert sales_row["role"] == ColumnRole.NUMERIC
    assert bool(sales_row["is_required"]) is True
    assert bool(sales_row["is_identifier"]) is False


def test_profile_columns_fails_when_required_column_missing() -> None:
    # Given
    frame = pd.DataFrame({"household_key": [10]})
    spec = DatasetSpec(name="t", required_cols=("SALES",))

    # When / Then
    with pytest.raises(DatasetSpecError):
        profile_columns(frame, table_name="t", spec=spec)

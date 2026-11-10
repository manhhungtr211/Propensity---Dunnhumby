from __future__ import annotations

import polars as pl
import pytest

from propensity.eda import (
    AnalysisColumnError,
    AnalysisConfigurationError,
    build_table_inventory,
    inspect_columns,
    summarize_keys,
    table_overview,
    validate_grain,
)
from propensity.schema import ColumnRole, DatasetSpec


def _transaction_spec() -> DatasetSpec:
    return DatasetSpec(
        name="transaction",
        entity_col="household_key",
        time_col="DAY",
        time_grain="day_index",
        grain_cols=("BASKET_ID", "PRODUCT_ID"),
        required_cols=("QUANTITY", "SALES_VALUE"),
        role_overrides={
            "BASKET_ID": ColumnRole.ID,
            "PRODUCT_ID": ColumnRole.ID,
        },
    )


def _transaction_frame() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "household_key": [1, 1, 2],
            "BASKET_ID": [100, 100, 101],
            "DAY": [1, 1, 3],
            "PRODUCT_ID": [10, 11, 10],
            "QUANTITY": [1, 2, 1],
            "SALES_VALUE": [5.0, 8.0, -10.0],
        }
    )


def test_table_overview_returns_table_identity_and_shape() -> None:
    # Given
    transactions = _transaction_frame()

    # When
    result = table_overview(transactions, _transaction_spec())

    # Then
    assert result.table_name == "transaction"
    assert result.n_rows == 3
    assert result.n_cols == 6


def test_inspect_columns_separates_dtype_and_semantic_role() -> None:
    # Given
    transactions = _transaction_frame()

    # When
    result = inspect_columns(transactions, _transaction_spec())

    # Then
    assert result.columns == [
        "position",
        "column",
        "physical_dtype",
        "semantic_role",
        "is_required",
        "is_grain",
    ]
    basket = result.filter(pl.col("column") == "BASKET_ID").row(0, named=True)
    assert basket == {
        "position": 1,
        "column": "BASKET_ID",
        "physical_dtype": "Int64",
        "semantic_role": "id",
        "is_required": True,
        "is_grain": True,
    }


def test_summarize_keys_excludes_time_and_numeric_columns() -> None:
    # Given
    transactions = _transaction_frame()

    # When
    result = summarize_keys(transactions, _transaction_spec())

    # Then
    assert result["column"].to_list() == [
        "household_key",
        "BASKET_ID",
        "PRODUCT_ID",
    ]
    assert result["n_unique"].to_list() == [2, 2, 2]
    assert "DAY" not in result["column"].to_list()


def test_summarize_keys_reports_null_key_without_dropping_rows() -> None:
    # Given
    transactions = pl.LazyFrame(
        {
            "household_key": [1, None],
            "BASKET_ID": [100, 101],
            "DAY": [1, 2],
            "PRODUCT_ID": [10, 20],
            "QUANTITY": [1, 1],
            "SALES_VALUE": [5.0, 8.0],
        }
    )

    # When
    result = summarize_keys(transactions, _transaction_spec())

    # Then
    entity = result.filter(pl.col("column") == "household_key").row(0, named=True)
    assert entity["n_rows"] == 2
    assert entity["n_valid"] == 1
    assert entity["n_null"] == 1
    assert entity["n_unique"] == 1


def test_validate_grain_reports_unique_candidate() -> None:
    # Given
    transactions = _transaction_frame()

    # When
    result = validate_grain(transactions, ("BASKET_ID", "PRODUCT_ID"))

    # Then
    assert result.n_rows == 3
    assert result.n_unique == 3
    assert result.n_duplicate_rows == 0
    assert result.is_unique


def test_validate_grain_reports_duplicate_candidate() -> None:
    # Given
    transactions = pl.LazyFrame(
        {
            "BASKET_ID": [100, 100, 101],
            "PRODUCT_ID": [10, 10, 20],
        }
    )

    # When
    result = validate_grain(transactions, ("BASKET_ID", "PRODUCT_ID"))

    # Then
    assert result.n_rows == 3
    assert result.n_unique == 2
    assert result.n_duplicate_rows == 1
    assert not result.is_unique


def test_validate_grain_rejects_empty_configuration() -> None:
    # Given
    transactions = _transaction_frame()

    # When / Then
    with pytest.raises(AnalysisConfigurationError, match="grain_cols"):
        validate_grain(transactions, ())


def test_validate_grain_rejects_missing_column() -> None:
    # Given
    transactions = _transaction_frame()

    # When / Then
    with pytest.raises(AnalysisColumnError, match="LINE_ID"):
        validate_grain(transactions, ("LINE_ID",))


def test_build_table_inventory_returns_notebook_contract() -> None:
    # Given
    transactions = _transaction_frame()

    # When
    result = build_table_inventory(transactions, _transaction_spec())

    # Then
    assert result.overview.n_rows == 3
    assert result.columns.height == 6
    assert result.keys.height == 3
    assert result.grain is not None
    assert result.grain.is_unique
    assert result.preview.height == 3

from __future__ import annotations

from pathlib import Path
from typing import Final

import polars as pl
import pytest

from propensity.io import scan_csv
from propensity.schema import DatasetSpec, DatasetSpecError
from propensity.schema.dunnhumby import TRANSACTION_SPEC

RAW_DATA_DIR: Final = Path(__file__).resolve().parents[1] / "data" / "raw"


def _transaction_spec() -> DatasetSpec:
    return TRANSACTION_SPEC


def _valid_transaction_frame(*, sales_value: float = 1.39) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "household_key": [2375],
            "DAY": [1],
            "BASKET_ID": [26984851472],
            "PRODUCT_ID": [1004906],
            "QUANTITY": [1],
            "SALES_VALUE": [sales_value],
        },
        schema={
            "household_key": pl.Int64,
            "DAY": pl.Int64,
            "BASKET_ID": pl.Int64,
            "PRODUCT_ID": pl.Int64,
            "QUANTITY": pl.Int64,
            "SALES_VALUE": pl.Float64,
        },
    )


def test_ac1_dataset_spec_describes_transaction_contract() -> None:
    # When
    spec = _transaction_spec()

    # Then
    assert spec.name == "transaction"
    assert spec.entity_col == "household_key"
    assert spec.time_col == "DAY"
    assert spec.time_grain == "day_index"
    assert spec.grain_cols == ("BASKET_ID", "PRODUCT_ID")


def test_ac2_real_transaction_csv_passes_validation() -> None:
    # Given
    transaction_path = RAW_DATA_DIR / "Dunnhumby" / "transaction_data.csv"
    transactions = scan_csv(transaction_path)

    # When
    spec = _transaction_spec()
    spec.validate(transactions)
    sample = transactions.select(spec.required_columns).head(5).collect()

    # Then
    assert sample.columns == list(spec.required_columns)
    assert sample.height == 5


def test_ac3_missing_required_column_fails_clearly() -> None:
    # Given
    spec = _transaction_spec()
    incomplete = _valid_transaction_frame().drop("BASKET_ID").lazy()

    # When / Then
    with pytest.raises(DatasetSpecError, match="BASKET_ID"):
        spec.validate(incomplete)


def test_ac4_unusual_business_value_is_not_a_schema_failure() -> None:
    # Given
    spec = _transaction_spec()
    unusual = _valid_transaction_frame(sales_value=-10.0)

    # When
    spec.validate(unusual.lazy())

    # Then
    assert unusual["SALES_VALUE"].item() == -10.0


def test_ac5_validation_does_not_mutate_data() -> None:
    # Given
    spec = _transaction_spec()
    transactions = _valid_transaction_frame()
    before = transactions.clone()

    # When
    spec.validate(transactions.lazy())

    # Then
    assert transactions.equals(before)

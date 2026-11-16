"""Behavioral tests for numeric distribution profiling (df.describe equivalent)."""

import pandas as pd
import pytest

from propensity.overview.models import NUMERIC_SUMMARY_CONTRACT
from propensity.overview.numeric import profile_numeric
from propensity.schema.spec import DatasetSpec


def test_profile_numeric_computes_distribution_metrics() -> None:
    # Given:
    # "SALES": [10.0, 20.0, 30.0, 40.0, 50.0]
    # "QUANTITY": [1, 2, None, 4, " "] (with null and blank)
    # "CODE": ["A", "B", "C", "D", "E"] (categorical, should not be included)
    frame = pd.DataFrame(
        {
            "SALES": [10.0, 20.0, 30.0, 40.0, 50.0],
            "QUANTITY": [1, 2, None, 4, " "],
            "CODE": ["A", "B", "C", "D", "E"],
        }
    )
    spec = DatasetSpec(
        name="sales_test",
        required_cols=("SALES", "QUANTITY"),
    )

    # When
    result = profile_numeric(frame, table_name="sales_test", spec=spec)

    # Then
    assert tuple(result.columns) == NUMERIC_SUMMARY_CONTRACT.columns
    assert len(result) == 2
    assert result["column"].tolist() == ["SALES", "QUANTITY"]

    sales_row = result[result["column"] == "SALES"].iloc[0]
    assert sales_row["count"] == 5
    assert sales_row["mean"] == pytest.approx(30.0)
    assert sales_row["std"] == pytest.approx(15.811, rel=1e-2)
    assert sales_row["min"] == 10.0
    assert sales_row["p25"] == 20.0
    assert sales_row["median"] == 30.0
    assert sales_row["p75"] == 40.0
    assert sales_row["max"] == 50.0

    qty_row = result[result["column"] == "QUANTITY"].iloc[0]
    assert qty_row["count"] == 3
    assert qty_row["min"] == 1.0
    assert qty_row["median"] == 2.0
    assert qty_row["max"] == 4.0


def test_profile_numeric_excludes_identifiers_and_time() -> None:
    # Given:
    # household_key is ID, DAY is TIME, SALES is NUMERIC
    frame = pd.DataFrame(
        {
            "household_key": [101, 102],
            "DAY": [1, 2],
            "SALES": [5.0, 10.0],
        }
    )
    spec = DatasetSpec(
        name="tx",
        entity_col="household_key",
        time_col="DAY",
        identifier_cols=("household_key",),
        ordinal_time_cols=("DAY",),
    )

    # When
    result = profile_numeric(frame, table_name="tx", spec=spec)

    # Then: only SALES is included, IDs and ordinal days are excluded!
    assert len(result) == 1
    assert result.iloc[0]["column"] == "SALES"


def test_profile_numeric_when_no_numeric_columns() -> None:
    # Given
    frame = pd.DataFrame({"name": ["a", "b"]})

    # When
    result = profile_numeric(frame, table_name="names")

    # Then
    assert tuple(result.columns) == NUMERIC_SUMMARY_CONTRACT.columns
    assert len(result) == 0

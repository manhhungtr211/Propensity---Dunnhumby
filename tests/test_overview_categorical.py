"""Behavioral tests for categorical distribution profiling (df.describe equivalent for categorical columns)."""

import pandas as pd
import pytest

from propensity.overview.categorical import profile_categorical
from propensity.overview.models import CATEGORICAL_SUMMARY_CONTRACT
from propensity.schema.spec import DatasetSpec


def test_profile_categorical_computes_distribution_metrics() -> None:
    # Given:
    # "DEPARTMENT": ["GROCERY", "GROCERY", "DRUG", "GROCERY", "DRUG"] -> count=5, unique=2, top="GROCERY", freq=3, top_pct=60.0
    # "BRAND": ["National", "Private", "National", None, " "] -> count=3, unique=2, top="National", freq=2, top_pct=66.67
    # "SALES": [10.0, 20.0, 30.0, 40.0, 50.0] -> numeric, should be excluded
    frame = pd.DataFrame(
        {
            "DEPARTMENT": ["GROCERY", "GROCERY", "DRUG", "GROCERY", "DRUG"],
            "BRAND": ["National", "Private", "National", None, " "],
            "SALES": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    spec = DatasetSpec(
        name="product_test",
        required_cols=("DEPARTMENT", "BRAND"),
    )

    # When
    result = profile_categorical(frame, table_name="product_test", spec=spec)

    # Then
    assert tuple(result.columns) == CATEGORICAL_SUMMARY_CONTRACT.columns
    assert len(result) == 2
    assert result["column"].tolist() == ["DEPARTMENT", "BRAND"]

    dept_row = result[result["column"] == "DEPARTMENT"].iloc[0]
    assert dept_row["count"] == 5
    assert dept_row["n_unique"] == 2
    assert dept_row["top"] == "GROCERY"
    assert dept_row["freq"] == 3
    assert dept_row["top_pct"] == pytest.approx(60.0)

    brand_row = result[result["column"] == "BRAND"].iloc[0]
    assert brand_row["count"] == 3
    assert brand_row["n_unique"] == 2
    assert brand_row["top"] == "National"
    assert brand_row["freq"] == 2
    assert brand_row["top_pct"] == pytest.approx(66.67, rel=1e-2)


def test_profile_categorical_excludes_identifiers_time_and_numeric() -> None:
    # Given:
    frame = pd.DataFrame(
        {
            "household_key": ["H1", "H2"],
            "DAY": [1, 2],
            "SALES": [5.0, 10.0],
            "STATUS": ["Active", "Active"],
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
    result = profile_categorical(frame, table_name="tx", spec=spec)

    # Then: only STATUS is included
    assert len(result) == 1
    assert result.iloc[0]["column"] == "STATUS"
    assert result.iloc[0]["top"] == "Active"
    assert result.iloc[0]["n_unique"] == 1


def test_profile_categorical_when_no_categorical_columns() -> None:
    # Given:
    frame = pd.DataFrame({"SALES": [10.0, 20.0]})

    # When
    result = profile_categorical(frame, table_name="sales_only")

    # Then:
    assert tuple(result.columns) == CATEGORICAL_SUMMARY_CONTRACT.columns
    assert len(result) == 0


def test_profile_categorical_all_missing_column() -> None:
    # Given:
    frame = pd.DataFrame({"CODE": [None, " ", None]})

    # When
    result = profile_categorical(frame, table_name="missing_test")

    # Then:
    assert len(result) == 1
    row = result.iloc[0]
    assert row["column"] == "CODE"
    assert row["count"] == 0
    assert row["n_unique"] == 0
    assert pd.isna(row["top"])
    assert pd.isna(row["freq"])
    assert pd.isna(row["top_pct"])


def test_profile_categorical_empty_frame() -> None:
    # Given:
    frame = pd.DataFrame()

    # When
    result = profile_categorical(frame, table_name="empty")

    # Then:
    assert tuple(result.columns) == CATEGORICAL_SUMMARY_CONTRACT.columns
    assert len(result) == 0

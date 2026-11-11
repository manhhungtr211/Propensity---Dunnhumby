from __future__ import annotations

from typing import Final

import polars as pl
import pytest

from propensity.eda.profiling import (
    CARDINALITY_COLUMNS,
    CATALOG_COLUMNS,
    build_feature_catalog,
    summarize_cardinality,
)
from propensity.schema.dunnhumby import (
    TRANSACTION_ANALYSIS_OVERRIDES,
    TRANSACTION_SPEC,
)

TRANSACTION_COLUMNS: Final = (
    "household_key",
    "BASKET_ID",
    "DAY",
    "PRODUCT_ID",
    "QUANTITY",
    "SALES_VALUE",
    "STORE_ID",
    "RETAIL_DISC",
    "TRANS_TIME",
    "WEEK_NO",
    "COUPON_DISC",
    "COUPON_MATCH_DISC",
)
@pytest.fixture
def transaction_frame() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "household_key": [1, 1, 2, 3],
            "BASKET_ID": [100, 100, 101, 102],
            "DAY": [1, 1, 3, 4],
            "PRODUCT_ID": [10, 11, 10, 12],
            "QUANTITY": [1, 2, 0, None],
            "SALES_VALUE": [5.0, 8.0, 0.0, 100.0],
            "STORE_ID": [7, 7, 8, 8],
            "RETAIL_DISC": [-1.0, 0.0, 0.0, -5.0],
            "TRANS_TIME": [930, 930, 2415, 45],
            "WEEK_NO": [1, 1, 1, 2],
            "COUPON_DISC": [0.0, 0.0, -0.5, 0.0],
            "COUPON_MATCH_DISC": [0.0, 0.0, 0.0, -0.5],
        }
    )


def test_feature_catalog_exposes_fixed_schema_for_every_column(
    transaction_frame: pl.LazyFrame,
) -> None:
    # Given / When
    result = build_feature_catalog(
        transaction_frame, TRANSACTION_SPEC, TRANSACTION_ANALYSIS_OVERRIDES
    )

    # Then
    assert result.columns == list(CATALOG_COLUMNS)
    assert result.height == len(TRANSACTION_COLUMNS)


@pytest.mark.parametrize(
    ("column", "structural_role", "analysis_kind"),
    [
        ("BASKET_ID", "id", "identifier"),
        ("SALES_VALUE", "numeric", "numeric"),
        ("DAY", "time", "ordinal_time"),
        ("TRANS_TIME", "time", "clock_time"),
        ("WEEK_NO", "time", "ordinal_time"),
    ],
)
def test_catalog_routes_by_semantics_instead_of_integer_storage(
    transaction_frame: pl.LazyFrame,
    column: str,
    structural_role: str,
    analysis_kind: str,
) -> None:
    # Given / When
    result = build_feature_catalog(
        transaction_frame, TRANSACTION_SPEC, TRANSACTION_ANALYSIS_OVERRIDES
    )

    # Then
    row = result.filter(pl.col("column") == column).row(0, named=True)
    assert row["structural_role"] == structural_role
    assert row["analysis_kind"] == analysis_kind


def test_transaction_declares_only_analysis_exceptions() -> None:
    assert TRANSACTION_ANALYSIS_OVERRIDES == {
        "TRANS_TIME": "clock_time",
        "WEEK_NO": "ordinal_time",
    }


def test_cardinality_returns_only_direct_counts() -> None:
    # Given
    frame = pl.LazyFrame({"value": [1, 1, None, 2]})

    # When
    result = summarize_cardinality(frame, ("value",))

    # Then
    assert result.columns == list(CARDINALITY_COLUMNS)
    assert result.row(0, named=True) == {
        "column": "value",
        "n_null": 1,
        "n_unique": 2,
    }


def test_all_null_cardinality_reports_zero_unique_values() -> None:
    # Given
    frame = pl.LazyFrame({"value": pl.Series([None, None], dtype=pl.Int64)})

    # When
    result = summarize_cardinality(frame, ("value",))

    # Then
    row = result.row(0, named=True)
    assert row["n_null"] == 2
    assert row["n_unique"] == 0

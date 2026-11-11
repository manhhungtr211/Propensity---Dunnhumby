from __future__ import annotations

from typing import Final

import polars as pl
import pytest

from propensity.eda.profiling import (
    NUMERIC_PROFILE_COLUMNS,
    TEMPORAL_PROFILE_COLUMNS,
    FeatureUnderstandingConfig,
    build_feature_catalog,
    build_feature_understanding,
    profile_numeric_features,
    profile_temporal_features,
)
from propensity.eda.quality import VALUE_QUALITY_COLUMNS, inspect_value_quality
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
NUMERIC_MEASURES: Final = (
    "QUANTITY",
    "SALES_VALUE",
    "RETAIL_DISC",
    "COUPON_DISC",
    "COUPON_MATCH_DISC",
)
TEMPORAL_FEATURES: Final = ("DAY", "TRANS_TIME", "WEEK_NO")
TEST_CONFIG: Final = FeatureUnderstandingConfig(
    analysis_overrides=TRANSACTION_ANALYSIS_OVERRIDES,
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


def test_data_quality_records_impossible_clock_time_without_mutation(
    transaction_frame: pl.LazyFrame,
) -> None:
    # Given
    catalog = build_feature_catalog(
        transaction_frame, TRANSACTION_SPEC, TRANSACTION_ANALYSIS_OVERRIDES
    )

    # When
    result = inspect_value_quality(transaction_frame, catalog)

    # Then
    assert result.columns == list(VALUE_QUALITY_COLUMNS)
    clock = result.filter(pl.col("column") == "TRANS_TIME").row(0, named=True)
    assert clock["impossible_values_found"] == ["2415"]
    assert transaction_frame.collect()["TRANS_TIME"].to_list() == [930, 930, 2415, 45]


def test_numeric_profile_contains_only_measures_and_descriptive_evidence(
    transaction_frame: pl.LazyFrame,
) -> None:
    # Given
    catalog = build_feature_catalog(
        transaction_frame, TRANSACTION_SPEC, TRANSACTION_ANALYSIS_OVERRIDES
    )

    # When
    result = profile_numeric_features(transaction_frame, catalog)

    # Then
    assert result.columns == list(NUMERIC_PROFILE_COLUMNS)
    assert set(result["column"]) == set(NUMERIC_MEASURES)
    coupon = result.filter(pl.col("column") == "COUPON_DISC").row(0, named=True)
    assert coupon["min"] == pytest.approx(-0.5)
    assert coupon["max"] == pytest.approx(0.0)


def test_temporal_profile_reports_ordinal_coverage_without_numeric_stats(
    transaction_frame: pl.LazyFrame,
) -> None:
    # Given
    catalog = build_feature_catalog(
        transaction_frame, TRANSACTION_SPEC, TRANSACTION_ANALYSIS_OVERRIDES
    )

    # When
    result = profile_temporal_features(transaction_frame, catalog)

    # Then
    assert result.columns == list(TEMPORAL_PROFILE_COLUMNS)
    assert set(result["column"]) == set(TEMPORAL_FEATURES)
    day = result.filter(pl.col("column") == "DAY").row(0, named=True)
    assert day["observed_min"] == "1"
    assert day["observed_max"] == "4"
    assert day["span_size"] == 4
    assert day["observed_periods"] == 3
    assert day["coverage_ratio"] == pytest.approx(0.75)
    clock = result.filter(pl.col("column") == "TRANS_TIME").row(0, named=True)
    assert clock["span_size"] is None
    assert clock["coverage_ratio"] is None


def test_combined_report_contains_all_contract_outputs(
    transaction_frame: pl.LazyFrame,
) -> None:
    # Given / When
    result = build_feature_understanding(
        transaction_frame, TRANSACTION_SPEC, TEST_CONFIG
    )

    # Then
    assert result.catalog.height == len(TRANSACTION_COLUMNS)
    assert result.cardinality.height == len(TRANSACTION_COLUMNS)
    assert set(result.numeric_profiles["column"]) == set(NUMERIC_MEASURES)
    assert set(result.temporal_profiles["column"]) == set(TEMPORAL_FEATURES)

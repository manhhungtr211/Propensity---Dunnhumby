from __future__ import annotations

import polars as pl
import pytest

from propensity.eda import (
    AnalysisColumnError,
    AnalysisConfigurationError,
    build_table_inventory,
    describe_entities,
    describe_quality,
    describe_temporal,
)
from propensity.schema import ColumnRole, DatasetSpec, DatasetSpecError


def _product_spec() -> DatasetSpec:
    return DatasetSpec(
        name="product",
        grain_cols=("PRODUCT_ID",),
        required_cols=("DEPARTMENT",),
        role_overrides={"PRODUCT_ID": ColumnRole.ID},
    )


def _product_frame() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "PRODUCT_ID": [100, 200],
            "DEPARTMENT": ["GROCERY", "PRODUCE"],
        }
    )


def test_dimension_spec_validates_without_entity_or_time() -> None:
    # Given
    spec = _product_spec()

    # When
    spec.validate(_product_frame())

    # Then
    assert spec.required_columns == ("PRODUCT_ID", "DEPARTMENT")


def test_required_columns_deduplicate_while_preserving_order() -> None:
    # Given / When
    spec = DatasetSpec(
        name="events",
        entity_col="customer_id",
        time_col="event_day",
        grain_cols=("customer_id", "event_id"),
        required_cols=("event_id", "value"),
    )

    # Then
    assert spec.required_columns == (
        "customer_id",
        "event_day",
        "event_id",
        "value",
    )


def test_time_grain_requires_time_column() -> None:
    # When / Then
    with pytest.raises(DatasetSpecError, match="time_col"):
        DatasetSpec(name="product", time_grain="day_index")


def test_entity_and_time_columns_must_differ() -> None:
    # When / Then
    with pytest.raises(DatasetSpecError, match="phải khác nhau"):
        DatasetSpec(name="events", entity_col="key", time_col="key")


def test_grain_columns_cannot_contain_an_empty_name() -> None:
    # When / Then
    with pytest.raises(DatasetSpecError, match="grain_cols"):
        DatasetSpec(name="product", grain_cols=("PRODUCT_ID", " "))


def test_missing_grain_column_fails_validation() -> None:
    # Given
    spec = _product_spec()
    missing_grain = pl.LazyFrame({"DEPARTMENT": ["GROCERY"]})

    # When / Then
    with pytest.raises(DatasetSpecError, match="PRODUCT_ID"):
        spec.validate(missing_grain)


def test_inventory_accepts_dimension_spec_without_entity_or_time() -> None:
    # Given
    spec = _product_spec()

    # When
    report = build_table_inventory(_product_frame(), spec)

    # Then
    assert report.overview.n_rows == 2
    assert report.keys.height == 1
    assert report.grain is not None
    assert report.grain.is_unique


def test_quality_accepts_dimension_spec_without_entity_or_time() -> None:
    # Given
    spec = _product_spec()

    # When
    report = describe_quality(_product_frame(), spec)

    # Then
    assert report["n_rows"] == 2
    assert report["n_duplicate_rows"] == 0


def test_entity_analysis_rejects_spec_without_entity_column() -> None:
    # Given
    spec = _product_spec()

    # When / Then
    with pytest.raises(AnalysisConfigurationError, match="entity_col"):
        describe_entities(_product_frame(), spec)


def test_temporal_analysis_rejects_spec_without_time_column() -> None:
    # Given
    spec = _product_spec()

    # When / Then
    with pytest.raises(AnalysisColumnError, match="time_col"):
        describe_temporal(_product_frame(), spec)

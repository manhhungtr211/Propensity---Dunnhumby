"""Behavioral tests for Dunnhumby dataset specifications."""

from pathlib import Path

import pandas as pd
import pytest

from propensity.schema.dunnhumby import (
    CAMPAIGN_DESC_SPEC,
    CAMPAIGN_TABLE_SPEC,
    CAUSAL_SPEC,
    COUPON_REDEMPT_SPEC,
    COUPON_SPEC,
    DEMOGRAPHIC_SPEC,
    DUNNHUMBY_CATALOG,
    PRODUCT_SPEC,
    TRANSACTION_SPEC,
)
from propensity.schema.spec import DatasetSpecError

RAW_DIR = Path("data/raw/Dunnhumby")


def test_catalog_contains_all_eight_tables() -> None:
    # Given / When / Then
    assert len(DUNNHUMBY_CATALOG) == 8
    expected_tables = {
        "transaction",
        "product",
        "demographic",
        "campaign_desc",
        "campaign_table",
        "coupon",
        "coupon_redempt",
        "causal",
    }
    assert set(DUNNHUMBY_CATALOG.keys()) == expected_tables


@pytest.mark.parametrize(
    "table_name",
    [
        "transaction",
        "product",
        "demographic",
        "campaign_desc",
        "campaign_table",
        "coupon",
        "coupon_redempt",
        "causal",
    ],
)
def test_each_table_matches_real_csv_header(table_name: str) -> None:
    # Given
    info = DUNNHUMBY_CATALOG[table_name]
    csv_path = RAW_DIR / info.filename
    assert csv_path.exists(), f"File {csv_path} does not exist"

    # When: read only the header (nrows=0)
    header_frame = pd.read_csv(csv_path, nrows=0)

    # Then: spec validation passes without error
    info.spec.validate(header_frame)
    for col in info.spec.required_columns:
        assert col in header_frame.columns, f"Column {col} missing in {info.filename}"


def test_candidate_grains_match_contracts() -> None:
    # Given / When / Then
    assert TRANSACTION_SPEC.grain_cols == ("BASKET_ID", "PRODUCT_ID")
    assert PRODUCT_SPEC.grain_cols == ("PRODUCT_ID",)
    assert DEMOGRAPHIC_SPEC.grain_cols == ("household_key",)
    assert CAMPAIGN_TABLE_SPEC.grain_cols == ("CAMPAIGN", "household_key")
    assert CAMPAIGN_DESC_SPEC.grain_cols == ("CAMPAIGN",)
    assert COUPON_SPEC.grain_cols == ("COUPON_UPC", "PRODUCT_ID", "CAMPAIGN")
    assert COUPON_REDEMPT_SPEC.grain_cols == (
        "household_key",
        "DAY",
        "COUPON_UPC",
        "CAMPAIGN",
    )
    assert CAUSAL_SPEC.grain_cols == ("PRODUCT_ID", "STORE_ID", "WEEK_NO")


def test_time_metadata_preserves_semantics() -> None:
    # Given / When / Then
    assert TRANSACTION_SPEC.ordinal_time_cols == ("DAY", "WEEK_NO")
    assert TRANSACTION_SPEC.clock_time_cols == ("TRANS_TIME",)
    assert CAMPAIGN_DESC_SPEC.interval_cols == (("START_DAY", "END_DAY"),)
    assert CAMPAIGN_DESC_SPEC.ordinal_time_cols == ("START_DAY", "END_DAY")
    assert CAUSAL_SPEC.ordinal_time_cols == ("WEEK_NO",)
    assert CAUSAL_SPEC.string_cols == ("display", "mailer")
    assert COUPON_REDEMPT_SPEC.ordinal_time_cols == ("DAY",)


def test_spec_validation_fails_when_required_column_is_missing() -> None:
    # Given
    info = DUNNHUMBY_CATALOG["transaction"]
    incomplete_frame = pd.DataFrame({"household_key": [1]})

    # When / Then
    with pytest.raises(DatasetSpecError) as exc_info:
        info.spec.validate(incomplete_frame)
    assert "cột bắt buộc không có trong dữ liệu" in str(exc_info.value)

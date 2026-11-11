from __future__ import annotations

from pathlib import Path
from typing import Final

import polars as pl
import pytest

from propensity.eda import build_table_inventory
from propensity.io import scan_csv
from propensity.schema import DatasetSpecError
from propensity.schema.dunnhumby import PRODUCT_SPEC

PRODUCT_PATH: Final = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "Dunnhumby" / "product.csv"
)


def test_product_spec_describes_dimension_contract() -> None:
    # When / Then
    assert PRODUCT_SPEC.name == "product"
    assert PRODUCT_SPEC.entity_col is None
    assert PRODUCT_SPEC.time_col is None
    assert PRODUCT_SPEC.grain_cols == ("PRODUCT_ID",)


def test_product_csv_passes_inventory_contract() -> None:
    # Given
    products = scan_csv(PRODUCT_PATH)

    # When
    inventory = build_table_inventory(products, PRODUCT_SPEC)

    # Then
    assert inventory.overview.n_rows == 92_353
    assert inventory.grain is not None
    assert inventory.grain.is_unique
    assert inventory.keys["column"].to_list() == ["PRODUCT_ID", "MANUFACTURER"]


def test_product_spec_rejects_missing_required_column() -> None:
    # Given
    incomplete = pl.LazyFrame(
        {
            "PRODUCT_ID": [1],
            "MANUFACTURER": [10],
            "DEPARTMENT": ["GROCERY"],
            "BRAND": ["Private"],
            "COMMODITY_DESC": ["BAKED BREAD/BUNS/ROLLS"],
            "SUB_COMMODITY_DESC": ["BREAD"],
        }
    )

    # When / Then
    with pytest.raises(DatasetSpecError, match="CURR_SIZE_OF_PRODUCT"):
        PRODUCT_SPEC.validate(incomplete)

"""Dataset contracts for Dunnhumby source tables."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import polars as pl

from propensity.schema.spec import ColumnRole, DatasetSpec

TRANSACTION_SPEC: Final = DatasetSpec(
    name="transaction",
    entity_col="household_key",
    time_col="DAY",
    time_grain="day_index",
    grain_cols=(
        "BASKET_ID",
        "PRODUCT_ID",
    ),
    required_cols=(
        "QUANTITY",
        "SALES_VALUE",
    ),
    role_overrides={
        "BASKET_ID": ColumnRole.ID,
        "PRODUCT_ID": ColumnRole.ID,
    },
)

TRANSACTION_SCHEMA_OVERRIDES: Final[Mapping[str, pl.DataType]] = MappingProxyType(
    {
        "COUPON_DISC": pl.Float64,
        "COUPON_MATCH_DISC": pl.Float64,
    }
)

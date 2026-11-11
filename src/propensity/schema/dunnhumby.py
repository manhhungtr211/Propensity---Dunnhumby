"""Dataset contracts for Dunnhumby source tables."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import polars as pl

from propensity.schema.features import AnalysisKind
from propensity.schema.spec import ColumnRole, DatasetSpec

TRANSACTION_SPEC: Final = DatasetSpec(
    name="transaction",
    entity_col="household_key",
    time_col="DAY",
    time_grain="day_index",
    grain_cols=("BASKET_ID", "PRODUCT_ID"),
    required_cols=("QUANTITY", "SALES_VALUE"),
    role_overrides={
        "BASKET_ID": ColumnRole.ID,
        "PRODUCT_ID": ColumnRole.ID,
        "STORE_ID": ColumnRole.ID,
        "TRANS_TIME": ColumnRole.TIME,
        "WEEK_NO": ColumnRole.TIME
    },
)

PRODUCT_SPEC: Final = DatasetSpec(
    name="product",
    grain_cols=("PRODUCT_ID",),
    required_cols=(
        "MANUFACTURER",
        "DEPARTMENT",
        "BRAND",
        "COMMODITY_DESC",
        "SUB_COMMODITY_DESC",
        "CURR_SIZE_OF_PRODUCT",
    ),
    role_overrides={
        "PRODUCT_ID": ColumnRole.ID,
        "MANUFACTURER": ColumnRole.ID,
    },
)

TRANSACTION_SCHEMA_OVERRIDES: Final[Mapping[str, pl.DataType]] = MappingProxyType(
    {
        "COUPON_DISC": pl.Float64,
        "COUPON_MATCH_DISC": pl.Float64,
    }
)

# CSV inference sees numeric-looking values first, while later rows contain letter codes.
CAUSAL_SCHEMA_OVERRIDES: Final[Mapping[str, pl.DataType]] = MappingProxyType(
    {
        "display": pl.String,
        "mailer": pl.String,
    }
)

# Only encodings whose analysis cannot be inferred from dtype and ColumnRole.
TRANSACTION_ANALYSIS_OVERRIDES: Final[Mapping[str, AnalysisKind]] = MappingProxyType(
    {
        "TRANS_TIME": AnalysisKind.CLOCK_TIME,
        "WEEK_NO": AnalysisKind.ORDINAL_TIME,
    }
)

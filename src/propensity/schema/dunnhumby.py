"""Dataset contracts for Dunnhumby source tables."""

from types import MappingProxyType
from typing import Final

from propensity.schema.features import AnalysisKind
from propensity.schema.spec import DatasetSpec

DUNNHUMBY_IDENTIFIER_COLS: Final = (
    "household_key",
    "BASKET_ID",
    "PRODUCT_ID",
    "STORE_ID",
)

TRANSACTION_SPEC: Final = DatasetSpec(
    name="transaction",
    entity_col="household_key",
    time_col="DAY",
    time_grain="day_index",
    grain_cols=("BASKET_ID", "PRODUCT_ID"),
    identifier_cols=DUNNHUMBY_IDENTIFIER_COLS,
    required_cols=("QUANTITY", "SALES_VALUE"),
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
    identifier_cols=("PRODUCT_ID", "MANUFACTURER"),
)

# Only encodings whose analysis cannot be inferred from dtype and ColumnRole.
TRANSACTION_ANALYSIS_OVERRIDES: Final = MappingProxyType(
    {
        "TRANS_TIME": AnalysisKind.CLOCK_TIME,
        "WEEK_NO": AnalysisKind.ORDINAL_TIME,
    }
)

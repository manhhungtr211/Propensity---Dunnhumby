"""Dataset contracts for Dunnhumby source tables."""

from typing import Final

from src.schema.spec import ColumnRole, DatasetSpec

TRANSACTION_SPEC: Final = DatasetSpec(
    name="transaction",
    entity_col="household_key",
    time_col="DAY",
    required_cols=(
        "BASKET_ID",
        "PRODUCT_ID",
        "QUANTITY",
        "SALES_VALUE",
    ),
    role_overrides={
        "BASKET_ID": ColumnRole.ID,
        "PRODUCT_ID": ColumnRole.ID,
    },
)

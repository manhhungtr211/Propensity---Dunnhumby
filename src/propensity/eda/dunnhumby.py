"""Dataset-specific EDA configuration for Dunnhumby tables."""

from typing import Final

from propensity.eda.transaction_models import TransactionProfileConfig

DUNNHUMBY_TRANSACTION_PROFILE_CONFIG: Final = TransactionProfileConfig(
    entity_col="household_key",
    basket_col="BASKET_ID",
    time_col="DAY",
    period_col="WEEK_NO",
    product_col="PRODUCT_ID",
    store_col="STORE_ID",
    quantity_col="QUANTITY",
    sales_col="SALES_VALUE",
    discount_cols=("RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC"),
)

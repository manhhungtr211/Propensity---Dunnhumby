"""Dataset contracts and catalog for all eight Dunnhumby source tables."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from propensity.schema.spec import DatasetSpec


@dataclass(frozen=True, slots=True)
class DunnhumbyTableInfo:
    """Catalog entry describing one Dunnhumby source table."""

    table: str
    filename: str
    spec: DatasetSpec
    row_meaning: str


DUNNHUMBY_IDENTIFIER_COLS: Final = (
    "household_key",
    "BASKET_ID",
    "PRODUCT_ID",
    "STORE_ID",
    "CAMPAIGN",
    "COUPON_UPC",
)

TRANSACTION_SPEC: Final = DatasetSpec(
    name="transaction",
    entity_col="household_key",
    time_col="DAY",
    time_grain="day_index",
    grain_cols=("BASKET_ID", "PRODUCT_ID"),
    identifier_cols=("household_key", "BASKET_ID", "PRODUCT_ID", "STORE_ID"),
    ordinal_time_cols=("DAY", "WEEK_NO"),
    clock_time_cols=("TRANS_TIME",),
    required_cols=(
        "household_key",
        "BASKET_ID",
        "DAY",
        "PRODUCT_ID",
        "QUANTITY",
        "SALES_VALUE",
        "STORE_ID",
        "TRANS_TIME",
        "WEEK_NO",
    ),
)

PRODUCT_SPEC: Final = DatasetSpec(
    name="product",
    grain_cols=("PRODUCT_ID",),
    identifier_cols=("PRODUCT_ID", "MANUFACTURER"),
    required_cols=(
        "PRODUCT_ID",
        "MANUFACTURER",
        "DEPARTMENT",
        "BRAND",
        "COMMODITY_DESC",
        "SUB_COMMODITY_DESC",
        "CURR_SIZE_OF_PRODUCT",
    ),
)

DEMOGRAPHIC_SPEC: Final = DatasetSpec(
    name="demographic",
    entity_col="household_key",
    grain_cols=("household_key",),
    identifier_cols=("household_key",),
    required_cols=("household_key", "AGE_DESC", "INCOME_DESC"),
)

CAMPAIGN_DESC_SPEC: Final = DatasetSpec(
    name="campaign_desc",
    grain_cols=("CAMPAIGN",),
    identifier_cols=("CAMPAIGN",),
    ordinal_time_cols=("START_DAY", "END_DAY"),
    interval_cols=(("START_DAY", "END_DAY"),),
    required_cols=("DESCRIPTION", "CAMPAIGN", "START_DAY", "END_DAY"),
)

CAMPAIGN_TABLE_SPEC: Final = DatasetSpec(
    name="campaign_table",
    entity_col="household_key",
    grain_cols=("CAMPAIGN", "household_key"),
    identifier_cols=("CAMPAIGN", "household_key"),
    required_cols=("DESCRIPTION", "household_key", "CAMPAIGN"),
)

COUPON_SPEC: Final = DatasetSpec(
    name="coupon",
    grain_cols=("COUPON_UPC", "PRODUCT_ID", "CAMPAIGN"),
    identifier_cols=("COUPON_UPC", "PRODUCT_ID", "CAMPAIGN"),
    required_cols=("COUPON_UPC", "PRODUCT_ID", "CAMPAIGN"),
)

COUPON_REDEMPT_SPEC: Final = DatasetSpec(
    name="coupon_redempt",
    entity_col="household_key",
    time_col="DAY",
    time_grain="day_index",
    grain_cols=("household_key", "DAY", "COUPON_UPC", "CAMPAIGN"),
    identifier_cols=("household_key", "COUPON_UPC", "CAMPAIGN"),
    ordinal_time_cols=("DAY",),
    required_cols=("household_key", "DAY", "COUPON_UPC", "CAMPAIGN"),
)

CAUSAL_SPEC: Final = DatasetSpec(
    name="causal",
    grain_cols=("PRODUCT_ID", "STORE_ID", "WEEK_NO"),
    identifier_cols=("PRODUCT_ID", "STORE_ID"),
    ordinal_time_cols=("WEEK_NO",),
    string_cols=("display", "mailer"),
    required_cols=("PRODUCT_ID", "STORE_ID", "WEEK_NO", "display", "mailer"),
)

DUNNHUMBY_CATALOG: Final = MappingProxyType(
    {
        "transaction": DunnhumbyTableInfo(
            table="transaction",
            filename="transaction_data.csv",
            spec=TRANSACTION_SPEC,
            row_meaning="store transaction line item",
        ),
        "product": DunnhumbyTableInfo(
            table="product",
            filename="product.csv",
            spec=PRODUCT_SPEC,
            row_meaning="product catalog and taxonomy",
        ),
        "demographic": DunnhumbyTableInfo(
            table="demographic",
            filename="hh_demographic.csv",
            spec=DEMOGRAPHIC_SPEC,
            row_meaning="household demographic profile",
        ),
        "campaign_desc": DunnhumbyTableInfo(
            table="campaign_desc",
            filename="campaign_desc.csv",
            spec=CAMPAIGN_DESC_SPEC,
            row_meaning="campaign duration and metadata",
        ),
        "campaign_table": DunnhumbyTableInfo(
            table="campaign_table",
            filename="campaign_table.csv",
            spec=CAMPAIGN_TABLE_SPEC,
            row_meaning="household campaign assignment",
        ),
        "coupon": DunnhumbyTableInfo(
            table="coupon",
            filename="coupon.csv",
            spec=COUPON_SPEC,
            row_meaning="coupon campaign product mapping",
        ),
        "coupon_redempt": DunnhumbyTableInfo(
            table="coupon_redempt",
            filename="coupon_redempt.csv",
            spec=COUPON_REDEMPT_SPEC,
            row_meaning="household coupon redemption event",
        ),
        "causal": DunnhumbyTableInfo(
            table="causal",
            filename="causal_data.csv",
            spec=CAUSAL_SPEC,
            row_meaning="weekly store product promotional exposure",
        ),
    }
)

"""Public entry points for Dunnhumby non-transaction table profiles."""

from propensity.eda.dunnhumby_causal import profile_causal_data
from propensity.eda.dunnhumby_demographic import profile_household_demographics
from propensity.eda.dunnhumby_marketing import (
    profile_campaign_descriptions,
    profile_campaign_table,
    profile_coupon_mappings,
    profile_coupon_redemptions,
)
from propensity.eda.dunnhumby_product import profile_products

__all__ = [
    "profile_campaign_descriptions",
    "profile_campaign_table",
    "profile_causal_data",
    "profile_coupon_mappings",
    "profile_coupon_redemptions",
    "profile_household_demographics",
    "profile_products",
]

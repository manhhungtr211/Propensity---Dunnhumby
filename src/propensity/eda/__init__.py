"""Exploratory Data Analysis (EDA) module."""

from propensity.config import DEFAULT_NA_TOKENS
from propensity.eda._errors import AnalysisColumnError, AnalysisConfigurationError
from propensity.eda.dunnhumby_tables import (
    profile_campaign_descriptions,
    profile_campaign_table,
    profile_causal_data,
    profile_coupon_mappings,
    profile_coupon_redemptions,
    profile_household_demographics,
    profile_products,
)
from propensity.eda.entity import EntityReport, describe_entities
from propensity.eda.inventory import (
    build_table_inventory,
    inspect_columns,
    preview,
    summarize_keys,
    table_overview,
    validate_grain,
)
from propensity.eda.inventory_models import GrainReport, TableInventory, TableOverview
from propensity.eda.profiling import (
    FeatureUnderstandingConfig,
    FeatureUnderstandingReport,
    build_feature_catalog,
    build_feature_understanding,
    profile_numeric_features,
    profile_temporal_features,
    summarize_cardinality,
)
from propensity.eda.quality import (
    QualityReport,
    describe_quality,
    inspect_value_quality,
)
from propensity.eda.temporal import TemporalReport, describe_temporal
from propensity.eda.transaction import (
    build_transaction_technical_report,
    profile_identifier_frequencies,
    profile_return_intervals,
    summarize_basket_distributions,
    summarize_household_distributions,
    summarize_period_activity,
)
from propensity.eda.transaction_models import (
    TransactionProfileConfig,
    TransactionTechnicalReport,
)
from propensity.eda.transaction_relationships import profile_transaction_relationships
from propensity.schema.spec import ColumnRole, DatasetSpec

__all__ = [
    "DEFAULT_NA_TOKENS",
    "AnalysisColumnError",
    "AnalysisConfigurationError",
    "ColumnRole",
    "DatasetSpec",
    "EntityReport",
    "FeatureUnderstandingConfig",
    "FeatureUnderstandingReport",
    "GrainReport",
    "QualityReport",
    "TableInventory",
    "TableOverview",
    "TemporalReport",
    "TransactionProfileConfig",
    "TransactionTechnicalReport",
    "build_feature_catalog",
    "build_feature_understanding",
    "build_table_inventory",
    "build_transaction_technical_report",
    "describe_entities",
    "describe_quality",
    "describe_temporal",
    "inspect_columns",
    "inspect_value_quality",
    "preview",
    "profile_campaign_descriptions",
    "profile_campaign_table",
    "profile_causal_data",
    "profile_coupon_mappings",
    "profile_coupon_redemptions",
    "profile_household_demographics",
    "profile_identifier_frequencies",
    "profile_numeric_features",
    "profile_products",
    "profile_return_intervals",
    "profile_temporal_features",
    "profile_transaction_relationships",
    "summarize_basket_distributions",
    "summarize_cardinality",
    "summarize_household_distributions",
    "summarize_keys",
    "summarize_period_activity",
    "table_overview",
    "validate_grain",
]

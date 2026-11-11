"""Public feature-understanding API."""

from propensity.eda.profiling.cardinality import summarize_cardinality
from propensity.eda.profiling.catalog import build_feature_catalog
from propensity.eda.profiling.models import (
    CARDINALITY_COLUMNS,
    CATALOG_COLUMNS,
    NUMERIC_PROFILE_COLUMNS,
    TEMPORAL_PROFILE_COLUMNS,
    FeatureUnderstandingConfig,
    FeatureUnderstandingReport,
)
from propensity.eda.profiling.numeric import profile_numeric_features
from propensity.eda.profiling.report import build_feature_understanding
from propensity.eda.profiling.temporal import profile_temporal_features

__all__ = [
    "CARDINALITY_COLUMNS",
    "CATALOG_COLUMNS",
    "NUMERIC_PROFILE_COLUMNS",
    "TEMPORAL_PROFILE_COLUMNS",
    "FeatureUnderstandingConfig",
    "FeatureUnderstandingReport",
    "build_feature_catalog",
    "build_feature_understanding",
    "profile_numeric_features",
    "profile_temporal_features",
    "summarize_cardinality",
]

"""High-level orchestration for notebook 01."""

from __future__ import annotations

import polars as pl

from propensity.eda.profiling.cardinality import summarize_cardinality
from propensity.eda.profiling.catalog import build_feature_catalog
from propensity.eda.profiling.models import (
    FeatureUnderstandingConfig,
    FeatureUnderstandingReport,
)
from propensity.eda.profiling.numeric import profile_numeric_features
from propensity.eda.profiling.temporal import profile_temporal_features
from propensity.schema import DatasetSpec


def build_feature_understanding(
    lf: pl.LazyFrame,
    spec: DatasetSpec,
    config: FeatureUnderstandingConfig,
) -> FeatureUnderstandingReport:
    """Build all structured feature-understanding evidence."""
    catalog = build_feature_catalog(lf, spec, config.analysis_overrides)
    columns = tuple(catalog["column"].to_list())
    return FeatureUnderstandingReport(
        catalog=catalog,
        cardinality=summarize_cardinality(lf, columns),
        numeric_profiles=profile_numeric_features(lf, catalog),
        temporal_profiles=profile_temporal_features(lf, catalog),
    )

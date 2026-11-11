"""Immutable contracts for feature-understanding reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import polars as pl

from propensity.schema import AnalysisKind

CATALOG_COLUMNS: Final = (
    "position",
    "column",
    "physical_dtype",
    "structural_role",
    "analysis_kind",
)
CARDINALITY_COLUMNS: Final = (
    "column",
    "n_null",
    "n_unique",
)
NUMERIC_PROFILE_COLUMNS: Final = (
    "column",
    "min",
    "max",
    "mean",
    "std",
    "skewness",
    "p01",
    "p25",
    "p50",
    "p75",
    "p99",
    "iqr",
    "lower_fence",
    "upper_fence",
    "n_below_lower_fence",
    "n_above_upper_fence",
)
TEMPORAL_PROFILE_COLUMNS: Final = (
    "column",
    "analysis_kind",
    "n_valid",
    "n_unique",
    "observed_min",
    "observed_max",
    "span_size",
    "observed_periods",
    "coverage_ratio",
)



@dataclass(frozen=True, slots=True)
class FeatureUnderstandingConfig:
    """Exceptional analysis routes that cannot be inferred from schema roles."""

    analysis_overrides: Mapping[str, AnalysisKind] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "analysis_overrides",
            MappingProxyType(dict(self.analysis_overrides)),
        )


@dataclass(frozen=True, slots=True)
class FeatureUnderstandingReport:
    """Structured evidence consumed by notebook 01."""

    catalog: pl.DataFrame
    cardinality: pl.DataFrame
    numeric_profiles: pl.DataFrame
    temporal_profiles: pl.DataFrame

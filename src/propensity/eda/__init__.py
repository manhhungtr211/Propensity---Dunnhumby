"""Exploratory Data Analysis (EDA) module."""

from propensity.config import DEFAULT_NA_TOKENS
from propensity.eda._errors import AnalysisColumnError, AnalysisConfigurationError
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
from propensity.eda.quality import QualityReport, describe_quality
from propensity.eda.temporal import TemporalReport, describe_temporal
from propensity.schema.spec import ColumnRole, DatasetSpec

__all__ = [
    "DEFAULT_NA_TOKENS",
    "AnalysisColumnError",
    "AnalysisConfigurationError",
    "ColumnRole",
    "DatasetSpec",
    "EntityReport",
    "GrainReport",
    "QualityReport",
    "TableInventory",
    "TableOverview",
    "TemporalReport",
    "build_table_inventory",
    "describe_entities",
    "describe_quality",
    "describe_temporal",
    "inspect_columns",
    "preview",
    "summarize_keys",
    "table_overview",
    "validate_grain",
]

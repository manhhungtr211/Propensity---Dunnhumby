"""Compatibility facade for the responsibility-based EDA modules."""

from propensity.eda import (
    EntityReport,
    GrainReport,
    QualityReport,
    TableInventory,
    TableOverview,
    TemporalReport,
    build_table_inventory,
    describe_entities,
    describe_quality,
    describe_temporal,
    inspect_columns,
    preview,
    summarize_keys,
    table_overview,
    validate_grain,
)
from propensity.eda.report import EDAReport, build_eda_report
from propensity.eda.target import TargetReport, describe_target

__all__ = [
    "EDAReport",
    "EntityReport",
    "GrainReport",
    "QualityReport",
    "TableInventory",
    "TableOverview",
    "TargetReport",
    "TemporalReport",
    "build_eda_report",
    "build_table_inventory",
    "describe_entities",
    "describe_quality",
    "describe_target",
    "describe_temporal",
    "inspect_columns",
    "preview",
    "summarize_keys",
    "table_overview",
    "validate_grain",
]

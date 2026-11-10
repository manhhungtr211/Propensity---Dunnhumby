"""Compatibility facade for the responsibility-based EDA modules."""

from src.eda.entity import EntityReport, describe_entities
from src.eda.inventory import (
    DatasetInventory,
    column_overview,
    describe_dataset,
    dtype_groups,
    preview,
    shape,
)
from src.eda.quality import QualityReport, describe_quality
from src.eda.report import EDAReport, build_eda_report
from src.eda.target import TargetReport, describe_target
from src.eda.temporal import TemporalReport, describe_temporal

__all__ = [
    "DatasetInventory",
    "EDAReport",
    "EntityReport",
    "QualityReport",
    "TargetReport",
    "TemporalReport",
    "build_eda_report",
    "column_overview",
    "describe_dataset",
    "describe_entities",
    "describe_quality",
    "describe_target",
    "describe_temporal",
    "dtype_groups",
    "preview",
    "shape",
]

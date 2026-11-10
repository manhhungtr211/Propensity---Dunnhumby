"""Optional orchestration for a combined EDA report."""

from __future__ import annotations

from typing import TypedDict

import polars as pl

from src.eda.entity import EntityReport, describe_entities
from src.eda.inventory import DatasetInventory, describe_dataset
from src.eda.quality import QualityReport, describe_quality
from src.eda.target import TargetReport, describe_target
from src.eda.temporal import TemporalReport, describe_temporal
from src.schema.spec import DatasetSpec


class EDAReport(TypedDict):
    """Combined report returned by :func:`build_eda_report`."""

    dataset_name: str
    dataset: DatasetInventory
    quality: QualityReport
    temporal: TemporalReport
    entity: EntityReport
    target: TargetReport | None


def build_eda_report(
    lf: pl.LazyFrame,
    spec: DatasetSpec,
) -> EDAReport:
    """Run all currently available EDA sections for one dataset."""
    target = describe_target(lf, spec) if spec.target_col is not None else None
    return {
        "dataset_name": spec.name,
        "dataset": describe_dataset(lf, spec),
        "quality": describe_quality(lf, spec),
        "temporal": describe_temporal(lf, spec),
        "entity": describe_entities(lf, spec),
        "target": target,
    }

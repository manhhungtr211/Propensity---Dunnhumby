"""Optional orchestration for a combined EDA report."""

from __future__ import annotations

from typing import TypedDict

import polars as pl

from propensity.eda.entity import EntityReport, describe_entities
from propensity.eda.inventory import build_table_inventory
from propensity.eda.inventory_models import TableInventory
from propensity.eda.quality import QualityReport, describe_quality
from propensity.eda.target import TargetReport, describe_target
from propensity.eda.temporal import TemporalReport, describe_temporal
from propensity.schema.spec import DatasetSpec


class EDAReport(TypedDict):
    """Combined report returned by :func:`build_eda_report`."""

    dataset_name: str
    dataset: TableInventory
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
        "dataset": build_table_inventory(lf, spec),
        "quality": describe_quality(lf, spec),
        "temporal": describe_temporal(lf, spec),
        "entity": describe_entities(lf, spec),
        "target": target,
    }

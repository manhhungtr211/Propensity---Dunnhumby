"""Resolve semantic column roles from business metadata and Polars dtypes."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import polars as pl

from propensity.schema.spec import ColumnRole, DatasetSpec


def infer_role(
    column: str,
    dtype: pl.DataType,
    spec: DatasetSpec,
) -> ColumnRole:
    """Infer one column role, prioritizing business semantics over dtype."""
    if column == spec.entity_col:
        return ColumnRole.ENTITY
    if column == spec.time_col:
        return ColumnRole.TIME
    if spec.target_col is not None and column == spec.target_col:
        return ColumnRole.TARGET

    override = spec.role_overrides.get(column)
    if override is not None:
        return override

    if dtype == pl.Boolean:
        return ColumnRole.BOOLEAN
    if dtype.is_temporal():
        return ColumnRole.DATETIME
    if dtype.is_numeric():
        return ColumnRole.NUMERIC
    if dtype.base_type() in (pl.String, pl.Categorical, pl.Enum):
        return ColumnRole.CATEGORICAL
    return ColumnRole.OTHER


def resolve_schema(
    lf: pl.LazyFrame,
    spec: DatasetSpec,
) -> Mapping[str, ColumnRole]:
    """Return immutable semantic roles for every column in a LazyFrame."""
    schema = lf.collect_schema()
    return MappingProxyType(
        {column: infer_role(column, dtype, spec) for column, dtype in schema.items()}
    )

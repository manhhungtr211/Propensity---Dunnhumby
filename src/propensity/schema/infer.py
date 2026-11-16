"""Resolve semantic column roles from business metadata and pandas dtypes."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from propensity.schema.spec import ColumnRole, DatasetSpec

DType: TypeAlias = np.dtype[np.generic] | pd.api.extensions.ExtensionDtype


def infer_role(
    column: str,
    dtype: DType,
    spec: DatasetSpec,
) -> ColumnRole:
    """Infer one column role, prioritizing business semantics over dtype."""
    if column == spec.entity_col:
        return ColumnRole.ENTITY
    if spec.target_col is not None and column == spec.target_col:
        return ColumnRole.TARGET

    if column in spec.identifier_cols:
        return ColumnRole.ID
    if column in spec.datetime_cols or ptypes.is_datetime64_any_dtype(dtype):
        return ColumnRole.DATETIME
    if (
        column == spec.time_col
        or column in spec.ordinal_time_cols
        or column in spec.clock_time_cols
        or any(column in endpoints for endpoints in spec.interval_cols)
    ):
        return ColumnRole.TIME
    if ptypes.is_bool_dtype(dtype):
        return ColumnRole.BOOLEAN
    if ptypes.is_numeric_dtype(dtype):
        return ColumnRole.NUMERIC
    if ptypes.is_string_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
        return ColumnRole.CATEGORICAL
    return ColumnRole.OTHER


def resolve_schema(
    frame: pd.DataFrame,
    spec: DatasetSpec,
) -> Mapping[str, ColumnRole]:
    """Return immutable semantic roles for every pandas column."""
    return MappingProxyType(
        {
            column: infer_role(column, dtype, spec)
            for column, dtype in frame.dtypes.items()
        }
    )

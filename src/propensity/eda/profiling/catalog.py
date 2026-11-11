"""Build the feature catalog used to route profiling computations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import assert_never

import polars as pl

from propensity.eda._errors import AnalysisConfigurationError
from propensity.eda.profiling.models import CATALOG_COLUMNS
from propensity.schema import AnalysisKind, ColumnRole, DatasetSpec, resolve_schema


def _default_analysis_kind(role: ColumnRole) -> AnalysisKind:
    match role:
        case ColumnRole.ENTITY | ColumnRole.ID:
            return AnalysisKind.IDENTIFIER
        case ColumnRole.NUMERIC | ColumnRole.TARGET:
            return AnalysisKind.NUMERIC
        case ColumnRole.CATEGORICAL | ColumnRole.BOOLEAN:
            return AnalysisKind.CATEGORICAL
        case ColumnRole.DATETIME:
            return AnalysisKind.DATETIME
        case ColumnRole.TIME:
            return AnalysisKind.ORDINAL_TIME
        case ColumnRole.OTHER:
            raise AnalysisConfigurationError(
                "ColumnRole.OTHER không suy ra được analysis_kind; hãy khai báo override"
            )
        case unreachable:
            assert_never(unreachable)


def build_feature_catalog(
    lf: pl.LazyFrame,
    spec: DatasetSpec,
    analysis_overrides: Mapping[str, AnalysisKind] | None = None,
) -> pl.DataFrame:
    """Combine inferred structural roles with exceptional analysis overrides."""
    schema = lf.collect_schema()
    structural_roles = resolve_schema(lf, spec)
    overrides = {} if analysis_overrides is None else analysis_overrides
    rows = []
    for position, column in enumerate(schema.names()):
        role = structural_roles[column]
        analysis_kind = overrides.get(column)
        if analysis_kind is None:
            analysis_kind = _default_analysis_kind(role)
        rows.append(
            (
                position,
                column,
                str(schema[column]),
                role.value,
                analysis_kind.value,
            )
        )
    return pl.DataFrame(
        rows,
        schema={
            "position": pl.UInt32,
            "column": pl.String,
            "physical_dtype": pl.String,
            "structural_role": pl.String,
            "analysis_kind": pl.String,
        },
        orient="row",
    ).select(CATALOG_COLUMNS)

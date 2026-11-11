"""Semantic dataset contracts and role inference."""

from propensity.schema.features import AnalysisKind
from propensity.schema.infer import infer_role, resolve_schema
from propensity.schema.spec import ColumnRole, DatasetSpec, DatasetSpecError

__all__ = [
    "AnalysisKind",
    "ColumnRole",
    "DatasetSpec",
    "DatasetSpecError",
    "infer_role",
    "resolve_schema",
]

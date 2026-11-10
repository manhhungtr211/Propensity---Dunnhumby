"""Semantic dataset contracts and role inference."""

from src.schema.infer import infer_role, resolve_schema
from src.schema.spec import ColumnRole, DatasetSpec, DatasetSpecError

__all__ = [
    "ColumnRole",
    "DatasetSpec",
    "DatasetSpecError",
    "infer_role",
    "resolve_schema",
]

"""Semantic contracts that physical DataFrame dtypes cannot express."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum, unique
from types import MappingProxyType

import polars as pl

from src.config import DEFAULT_NA_TOKENS


@unique
class ColumnRole(StrEnum):
    """Closed set of semantic roles used by schema-aware analysis."""

    ENTITY = "entity"
    TIME = "time"
    TARGET = "target"
    ID = "id"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class DatasetSpecError(ValueError):
    """Raised when a dataset specification is internally inconsistent."""

    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Semantic dataset contract layered on top of physical Polars dtypes."""

    name: str
    entity_col: str
    time_col: str
    target_col: str | None = None
    time_grain: str = "1d"
    role_overrides: Mapping[str, ColumnRole] = field(default_factory=dict)
    required_cols: tuple[str, ...] = ()
    na_tokens: tuple[str, ...] = DEFAULT_NA_TOKENS

    def __post_init__(self) -> None:
        if self.entity_col == self.time_col:
            raise DatasetSpecError("entity_col và time_col phải khác nhau")
        object.__setattr__(
            self,
            "role_overrides",
            MappingProxyType(dict(self.role_overrides)),
        )

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return required raw columns without requiring an optional target."""
        return tuple(
            dict.fromkeys(
                (
                    self.entity_col,
                    self.time_col,
                    *self.required_cols,
                )
            )
        )

    def validate(self, lf: pl.LazyFrame) -> None:
        """Raise when a required raw column is absent from the dataset."""
        schema = lf.collect_schema()
        available = set(schema.names())
        missing = [
            column for column in self.required_columns if column not in available
        ]
        if missing:
            raise DatasetSpecError(
                f"[{self.name}] cột bắt buộc không có trong dữ liệu: {missing}"
            )

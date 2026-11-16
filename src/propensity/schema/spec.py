"""Small semantic configuration for pandas datasets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique

import pandas as pd

from propensity.config import DEFAULT_NA_TOKENS


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


class DatasetSpecError(ValueError):
    """Raised when a dataset specification is internally inconsistent."""

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Columns needed to load and interpret one pandas dataset."""

    name: str
    entity_col: str | None = None
    time_col: str | None = None
    target_col: str | None = None
    time_grain: str | None = None
    grain_cols: tuple[str, ...] = ()
    identifier_cols: tuple[str, ...] = ()
    datetime_cols: tuple[str, ...] = ()
    required_cols: tuple[str, ...] = ()
    na_tokens: tuple[str, ...] = DEFAULT_NA_TOKENS

    def __post_init__(self) -> None:
        if self.entity_col is not None and self.entity_col == self.time_col:
            raise DatasetSpecError("entity_col và time_col phải khác nhau")
        if self.time_grain is not None and self.time_col is None:
            raise DatasetSpecError("time_grain yêu cầu time_col")
        if any(not column.strip() for column in self.grain_cols):
            raise DatasetSpecError("grain_cols không được chứa tên cột rỗng")
        if any(not column.strip() for column in self.identifier_cols):
            raise DatasetSpecError("identifier_cols không được chứa tên cột rỗng")
        if any(not column.strip() for column in self.datetime_cols):
            raise DatasetSpecError("datetime_cols không được chứa tên cột rỗng")

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return required raw columns without requiring an optional target."""
        semantic_columns = tuple(
            column for column in (self.entity_col, self.time_col) if column is not None
        )
        return tuple(
            dict.fromkeys(
                (
                    *semantic_columns,
                    *self.grain_cols,
                    *self.identifier_cols,
                    *self.datetime_cols,
                    *self.required_cols,
                )
            )
        )

    def validate(self, frame: pd.DataFrame) -> None:
        """Raise when a required raw column is absent from the dataset."""
        available = set(frame.columns)
        missing = [
            column for column in self.required_columns if column not in available
        ]
        if missing:
            raise DatasetSpecError(
                f"[{self.name}] cột bắt buộc không có trong dữ liệu: {missing}"
            )

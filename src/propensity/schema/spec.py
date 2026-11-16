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
    string_cols: tuple[str, ...] = ()
    ordinal_time_cols: tuple[str, ...] = ()
    clock_time_cols: tuple[str, ...] = ()
    interval_cols: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.entity_col is not None and self.entity_col == self.time_col:
            raise DatasetSpecError("entity_col và time_col phải khác nhau")
        if self.time_grain is not None and self.time_col is None:
            raise DatasetSpecError("time_grain yêu cầu time_col")
        for field, value in (
            ("name", self.name),
            ("entity_col", self.entity_col),
            ("time_col", self.time_col),
            ("target_col", self.target_col),
            ("time_grain", self.time_grain),
        ):
            if value is not None and not value.strip():
                raise DatasetSpecError(f"{field} must not be blank")
        for field, columns in (
            ("grain_cols", self.grain_cols),
            ("identifier_cols", self.identifier_cols),
            ("datetime_cols", self.datetime_cols),
            ("required_cols", self.required_cols),
            ("string_cols", self.string_cols),
            ("ordinal_time_cols", self.ordinal_time_cols),
            ("clock_time_cols", self.clock_time_cols),
        ):
            if any(not column.strip() for column in columns):
                raise DatasetSpecError(f"{field} must not contain blank columns")
            if len(set(columns)) != len(columns):
                raise DatasetSpecError(f"{field} must not contain duplicate columns")
        time_encodings = (
            set(self.datetime_cols),
            set(self.ordinal_time_cols),
            set(self.clock_time_cols),
        )
        for index, columns in enumerate(time_encodings):
            for other in time_encodings[index + 1 :]:
                overlap = columns & other
                if overlap:
                    raise DatasetSpecError(
                        f"Conflicting time encodings: {sorted(overlap)}"
                    )
        if set(self.string_cols) & set(self.datetime_cols):
            raise DatasetSpecError("string_cols and datetime_cols must not overlap")
        for endpoints in self.interval_cols:
            if len(endpoints) != 2 or any(not column.strip() for column in endpoints):
                raise DatasetSpecError("Intervals require two nonblank endpoints")
            if endpoints[0] == endpoints[1]:
                raise DatasetSpecError("Interval endpoints must differ")
        if len(set(self.interval_cols)) != len(self.interval_cols):
            raise DatasetSpecError("interval_cols must not contain duplicate pairs")

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
                    *self.string_cols,
                    *self.ordinal_time_cols,
                    *self.clock_time_cols,
                    *(column for pair in self.interval_cols for column in pair),
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

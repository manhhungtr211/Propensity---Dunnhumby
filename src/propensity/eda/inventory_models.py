"""Immutable result models for structural data inventory."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, slots=True)
class TableOverview:
    """Table-level shape metadata."""

    table_name: str
    n_rows: int
    n_cols: int


@dataclass(frozen=True, slots=True)
class GrainReport:
    """Uniqueness evidence for a candidate row grain."""

    grain_cols: tuple[str, ...]
    n_rows: int
    n_unique: int

    @property
    def n_duplicate_rows(self) -> int:
        """Return rows beyond the unique grain combinations."""
        return self.n_rows - self.n_unique

    @property
    def is_unique(self) -> bool:
        """Return whether the candidate grain identifies every row."""
        return self.n_duplicate_rows == 0


@dataclass(frozen=True, slots=True)
class TableInventory:
    """Structural evidence consumed by the data-inventory notebook."""

    overview: TableOverview
    columns: pl.DataFrame
    keys: pl.DataFrame
    grain: GrainReport | None
    preview: pl.DataFrame

"""CSV input boundary for lazy Polars pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import polars as pl


def scan_csv(
    path: str | Path,
    *,
    schema_overrides: Mapping[str, pl.DataType] | None = None,
) -> pl.LazyFrame:
    """Return a lazy CSV scan for one existing local file."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    overrides = None if schema_overrides is None else dict(schema_overrides)
    return pl.scan_csv(csv_path, schema_overrides=overrides)

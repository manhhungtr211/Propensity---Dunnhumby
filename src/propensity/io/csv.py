"""CSV input boundary for pandas pipelines."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from propensity.schema.spec import DatasetSpec


def read_csv(
    path: str | Path,
    *,
    spec: DatasetSpec | None = None,
) -> pd.DataFrame:
    """Load one CSV and apply the dataset's simple semantic configuration."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    dtype = None
    parse_dates = None
    if spec is not None:
        dtype = {column: "Int64" for column in spec.identifier_cols}
        parse_dates = list(spec.datetime_cols) or None

    frame = pd.read_csv(csv_path, dtype=dtype, parse_dates=parse_dates)
    if spec is not None:
        spec.validate(frame)
    return frame

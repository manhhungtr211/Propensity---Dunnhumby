"""Customizable table and dataset preview functions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import pandas as pd

from propensity.schema.dunnhumby import DUNNHUMBY_CATALOG


def preview_table(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    n_rows: int = 5,
    mode: Literal["head", "tail", "sample"] = "head",
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Return a customized preview slice of a DataFrame by rows and columns.

    Parameters
    ----------
    frame : pd.DataFrame
        Source DataFrame to preview.
    columns : Sequence[str] | None, optional
        Specific subset of columns to display. If None, displays all columns.
        Columns not found in the DataFrame are silently ignored.
    n_rows : int, default 5
        Number of rows to preview.
    mode : {"head", "tail", "sample"}, default "head"
        Whether to take the first n rows ("head"), last n rows ("tail"),
        or a random sample of n rows ("sample").
    random_state : int | None, default 42
        Seed for reproducibility when mode is "sample".

    Returns
    -------
    pd.DataFrame
        Preview subset of the specified rows and columns.
    """
    if len(frame) == 0:
        target_cols = [c for c in columns if c in frame.columns] if columns else list(frame.columns)
        return frame.loc[:, target_cols]

    if mode == "head":
        sliced = frame.head(n_rows)
    elif mode == "tail":
        sliced = frame.tail(n_rows)
    elif mode == "sample":
        n_sample = min(n_rows, len(frame))
        sliced = frame.sample(n=n_sample, random_state=random_state)
    else:
        raise ValueError(f"Unknown preview mode: '{mode}'. Expected 'head', 'tail', or 'sample'.")

    if columns is not None:
        valid_cols = [c for c in columns if c in sliced.columns]
        sliced = sliced.loc[:, valid_cols]

    return sliced.copy()


def preview_dunnhumby_tables(
    raw_dir: Path | str,
    *,
    tables: Sequence[str] | None = None,
    n_rows: int = 5,
    columns: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
    mode: Literal["head", "tail", "sample"] = "head",
    random_state: int | None = 42,
) -> dict[str, pd.DataFrame]:
    """Preview multiple Dunnhumby raw CSV tables with customized rows and columns.

    For mode='head', reads only the required rows from disk without scanning entire files.

    Parameters
    ----------
    raw_dir : Path | str
        Directory containing the raw CSV files.
    tables : Sequence[str] | None, optional
        Subset of tables to preview. Defaults to all tables in DUNNHUMBY_CATALOG.
    n_rows : int, default 5
        Number of rows to preview per table.
    columns : Mapping[str, Sequence[str]] | Sequence[str] | None, optional
        Subset of columns. Either a mapping of {table_name: [cols]} or a single
        sequence of column names applied across matching tables.
    mode : {"head", "tail", "sample"}, default "head"
        Sampling strategy: "head", "tail", or "sample".
    random_state : int | None, default 42
        Seed for reproducibility when mode is "sample".

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping each table name to its preview DataFrame.
    """
    raw_path = Path(raw_dir)
    target_tables = tuple(tables) if tables is not None else tuple(DUNNHUMBY_CATALOG.keys())

    # Map possible aliases
    aliases = {
        "hh_demographic": "demographic",
        "transaction_data": "transaction",
        "causal_data": "causal",
    }
    canonical_tables = tuple(aliases.get(t, t) for t in target_tables)

    results: dict[str, pd.DataFrame] = {}

    for tbl in canonical_tables:
        if tbl not in DUNNHUMBY_CATALOG:
            raise KeyError(f"Unknown Dunnhumby table: '{tbl}'")

        info = DUNNHUMBY_CATALOG[tbl]
        file_path = raw_path / info.filename
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Resolve target columns for this specific table
        table_cols: Sequence[str] | None = None
        if isinstance(columns, Mapping):
            table_cols = columns.get(tbl)
        elif isinstance(columns, Sequence):
            table_cols = columns

        # Efficient head reading directly from CSV without full file scan
        if mode == "head":
            frame = pd.read_csv(file_path, nrows=n_rows)
            preview_df = preview_table(frame, columns=table_cols, n_rows=n_rows, mode="head")
        else:
            # For tail or sample, read entire file with spec
            from propensity.io.csv import read_csv

            frame = read_csv(file_path, spec=info.spec)
            preview_df = preview_table(
                frame,
                columns=table_cols,
                n_rows=n_rows,
                mode=mode,
                random_state=random_state,
            )

        results[tbl] = preview_df

    return results

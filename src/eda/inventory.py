"""Dataset structure and column inventory."""

from __future__ import annotations

from typing import TypedDict

import polars as pl

from src.config import DEFAULT_NA_TOKENS
from src.schema.infer import resolve_schema
from src.schema.spec import ColumnRole, DatasetSpec

_OVERVIEW_SCHEMA = {
    "column": pl.String,
    "dtype": pl.String,
    "role": pl.String,
    "n_missing": pl.Int64,
    "missing_frac": pl.Float64,
    "n_unique_approx": pl.Int64,
}


class DatasetInventory(TypedDict, total=False):
    """Dataset-level structure returned by :func:`describe_dataset`."""

    n_rows: int
    n_cols: int
    dtype_groups: dict[str, list[str]]
    overview: pl.DataFrame
    preview: pl.DataFrame
    unassigned: list[str]


def _missing_expr(name: str, dtype: pl.DataType, tokens: tuple[str, ...]) -> pl.Expr:
    """Return an expression counting missing values in one column."""
    column = pl.col(name)
    condition = column.is_null()
    if dtype.is_float():
        condition = condition | column.is_nan().fill_null(False)
    if dtype == pl.String:
        normalized = column.str.strip_chars().str.to_lowercase()
        condition = condition | normalized.is_in(list(tokens)).fill_null(False)
    return condition.sum().alias(f"{name}||miss")


def _groups_from_schema(schema: pl.Schema) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "numeric": [],
        "string": [],
        "categorical": [],
        "temporal": [],
        "boolean": [],
        "other": [],
    }
    for name, dtype in schema.items():
        if dtype.is_numeric():
            groups["numeric"].append(name)
            continue
        if dtype == pl.String:
            groups["string"].append(name)
            continue
        if dtype in (pl.Categorical, pl.Enum):
            groups["categorical"].append(name)
            continue
        if dtype.is_temporal():
            groups["temporal"].append(name)
            continue
        if dtype == pl.Boolean:
            groups["boolean"].append(name)
            continue
        groups["other"].append(name)
    return groups


def _profile(
    lf: pl.LazyFrame,
    spec: DatasetSpec | None = None,
) -> tuple[int, pl.Schema, pl.DataFrame]:
    """Scan once and return row count, physical schema, and column overview."""
    tokens = spec.na_tokens if spec is not None else DEFAULT_NA_TOKENS
    schema = lf.collect_schema()
    columns = schema.names()
    flat_columns = [column for column in columns if not schema[column].is_nested()]
    roles = resolve_schema(lf, spec) if spec is not None else {}

    statistics = (
        lf.select(
            [pl.len().alias("__n")]
            + [_missing_expr(column, schema[column], tokens) for column in columns]
            + [
                pl.col(column).approx_n_unique().alias(f"{column}||nuniq")
                for column in flat_columns
            ]
        )
        .collect()
        .row(0, named=True)
    )
    n_rows = int(statistics["__n"])
    rows = [
        {
            "column": column,
            "dtype": str(schema[column]),
            "role": roles[column].value if spec is not None else None,
            "n_missing": statistics[f"{column}||miss"],
            "missing_frac": (statistics[f"{column}||miss"] / n_rows if n_rows else 0.0),
            "n_unique_approx": statistics.get(f"{column}||nuniq"),
        }
        for column in columns
    ]
    return n_rows, schema, pl.DataFrame(rows, schema=_OVERVIEW_SCHEMA)


def shape(lf: pl.LazyFrame) -> tuple[int, int]:
    """Return dataset row and column counts."""
    return lf.select(pl.len()).collect().item(), len(lf.collect_schema())


def preview(lf: pl.LazyFrame, n: int = 5) -> pl.DataFrame:
    """Collect the first ``n`` rows for inspection."""
    return lf.head(n).collect()


def dtype_groups(lf: pl.LazyFrame) -> dict[str, list[str]]:
    """Group column names by physical Polars dtype family."""
    return _groups_from_schema(lf.collect_schema())


def column_overview(
    lf: pl.LazyFrame,
    spec: DatasetSpec | None = None,
) -> pl.DataFrame:
    """Return one structural summary row per column."""
    return _profile(lf, spec)[2]


def describe_dataset(
    lf: pl.LazyFrame,
    spec: DatasetSpec | None = None,
) -> DatasetInventory:
    """Build a reusable structural inventory for one dataset."""
    n_rows, schema, overview = _profile(lf, spec)
    result: DatasetInventory = {
        "n_rows": n_rows,
        "n_cols": len(schema),
        "dtype_groups": _groups_from_schema(schema),
        "overview": overview,
        "preview": preview(lf),
    }
    if spec is not None:
        roles = resolve_schema(lf, spec)
        result["unassigned"] = [
            column for column, role in roles.items() if role is ColumnRole.OTHER
        ]
    return result

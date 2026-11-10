"""Structural inventory for understanding a table and its row grain."""

from __future__ import annotations

import polars as pl

from propensity.eda._errors import AnalysisColumnError, AnalysisConfigurationError
from propensity.eda.inventory_models import GrainReport, TableInventory, TableOverview
from propensity.schema.infer import resolve_schema
from propensity.schema.spec import ColumnRole, DatasetSpec


def preview(lf: pl.LazyFrame, n: int = 5) -> pl.DataFrame:
    """Collect at most the first ``n`` rows without transforming values."""
    return lf.head(n).collect()


def table_overview(lf: pl.LazyFrame, spec: DatasetSpec) -> TableOverview:
    """Return table identity and physical shape."""
    return TableOverview(
        table_name=spec.name,
        n_rows=lf.select(pl.len()).collect().item(),
        n_cols=len(lf.collect_schema()),
    )


def inspect_columns(lf: pl.LazyFrame, spec: DatasetSpec) -> pl.DataFrame:
    """Describe physical and declared semantic metadata for every column."""
    schema = lf.collect_schema()
    roles = resolve_schema(lf, spec)
    required = frozenset(spec.required_columns)
    grain = frozenset(spec.grain_cols)
    rows = [
        {
            "position": position,
            "column": column,
            "physical_dtype": str(dtype),
            "semantic_role": roles[column].value,
            "is_required": column in required,
            "is_grain": column in grain,
        }
        for position, (column, dtype) in enumerate(schema.items())
    ]
    return pl.DataFrame(
        rows,
        schema={
            "position": pl.UInt32,
            "column": pl.String,
            "physical_dtype": pl.String,
            "semantic_role": pl.String,
            "is_required": pl.Boolean,
            "is_grain": pl.Boolean,
        },
    )


def summarize_keys(lf: pl.LazyFrame, spec: DatasetSpec) -> pl.DataFrame:
    """Return exact cardinality for declared entity and ID columns."""
    roles = resolve_schema(lf, spec)
    key_columns = [
        column
        for column, role in roles.items()
        if role in (ColumnRole.ENTITY, ColumnRole.ID)
    ]
    result_schema = {
        "column": pl.String,
        "semantic_role": pl.String,
        "n_rows": pl.Int64,
        "n_valid": pl.Int64,
        "n_null": pl.Int64,
        "n_unique": pl.Int64,
    }
    if not key_columns:
        return pl.DataFrame(schema=result_schema)

    expressions: list[pl.Expr] = [pl.len().alias("__n_rows")]
    for column in key_columns:
        expressions.extend(
            (
                pl.col(column).count().alias(f"{column}||valid"),
                pl.col(column).null_count().alias(f"{column}||null"),
                pl.col(column).drop_nulls().n_unique().alias(f"{column}||unique"),
            )
        )
    statistics = lf.select(expressions).collect().row(0, named=True)
    return pl.DataFrame(
        [
            {
                "column": column,
                "semantic_role": roles[column].value,
                "n_rows": statistics["__n_rows"],
                "n_valid": statistics[f"{column}||valid"],
                "n_null": statistics[f"{column}||null"],
                "n_unique": statistics[f"{column}||unique"],
            }
            for column in key_columns
        ],
        schema=result_schema,
    )


def validate_grain(
    lf: pl.LazyFrame,
    grain_cols: tuple[str, ...],
) -> GrainReport:
    """Measure whether a candidate column combination identifies every row."""
    if not grain_cols:
        raise AnalysisConfigurationError("grain_cols không được để trống")

    available = frozenset(lf.collect_schema().names())
    missing = [column for column in grain_cols if column not in available]
    if missing:
        raise AnalysisColumnError(missing[0], "không tồn tại trong dữ liệu")

    statistics = (
        lf.select(
            pl.len().alias("n_rows"),
            pl.struct(list(grain_cols)).n_unique().alias("n_unique"),
        )
        .collect()
        .row(0, named=True)
    )
    return GrainReport(
        grain_cols=grain_cols,
        n_rows=int(statistics["n_rows"]),
        n_unique=int(statistics["n_unique"]),
    )


def build_table_inventory(
    lf: pl.LazyFrame,
    spec: DatasetSpec,
) -> TableInventory:
    """Build the structural evidence displayed by the inventory notebook."""
    spec.validate(lf)
    grain = validate_grain(lf, spec.grain_cols) if spec.grain_cols else None
    return TableInventory(
        overview=table_overview(lf, spec),
        columns=inspect_columns(lf, spec),
        keys=summarize_keys(lf, spec),
        grain=grain,
        preview=preview(lf),
    )

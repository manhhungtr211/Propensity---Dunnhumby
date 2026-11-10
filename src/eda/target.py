"""Target-label distribution checks."""

from __future__ import annotations

from typing import TypedDict

import polars as pl

from src.config import DEFAULT_NA_TOKENS
from src.eda.quality import _missing_condition
from src.eda.temporal import AnalysisColumnError
from src.schema.spec import DatasetSpec


class TargetReport(TypedDict):
    """Result returned by :func:`describe_target`."""

    target_col: str
    n_records: int
    n_valid_target: int
    n_missing_target: int
    missing_target_pct: float
    n_classes: int
    majority_class_pct: float
    minority_class_pct: float
    imbalance_ratio: float | None
    distribution: pl.DataFrame


def describe_target(
    lf: pl.LazyFrame,
    spec: DatasetSpec | None = None,
    *,
    target_col: str | None = None,
) -> TargetReport:
    """Summarize a categorical label or event target, excluding missing values."""
    if spec is not None:
        target_col = spec.target_col
        tokens = spec.na_tokens
    else:
        tokens = DEFAULT_NA_TOKENS
    if target_col is None:
        raise AnalysisColumnError("target_col", "chưa được cấu hình")

    schema = lf.collect_schema()
    if target_col not in schema:
        raise AnalysisColumnError(
            target_col,
            "không tồn tại trong dữ liệu",
        )

    missing = _missing_condition(target_col, schema[target_col], tokens)
    prepared = lf.select(
        pl.col(target_col),
        missing.alias("__missing_target"),
    )
    totals = (
        prepared.select(
            pl.len().alias("n_records"),
            pl.col("__missing_target").sum().alias("n_missing_target"),
        )
        .collect()
        .row(0, named=True)
    )
    n_records = int(totals["n_records"])
    n_missing_target = int(totals["n_missing_target"])
    n_valid_target = n_records - n_missing_target

    distribution = (
        prepared.filter(~pl.col("__missing_target"))
        .group_by(target_col)
        .agg(pl.len().alias("n_records"))
        .with_columns(
            (100.0 * pl.col("n_records") / n_valid_target).alias("pct")
            if n_valid_target > 0
            else pl.lit(0.0).alias("pct"),
        )
        .sort("n_records", descending=True)
        .collect()
    )
    if distribution.is_empty():
        majority_pct = 0.0
        minority_pct = 0.0
        imbalance_ratio = None
    else:
        majority_count = int(distribution["n_records"].max())
        minority_count = int(distribution["n_records"].min())
        majority_pct = 100.0 * majority_count / n_valid_target
        minority_pct = 100.0 * minority_count / n_valid_target
        imbalance_ratio = majority_count / minority_count

    return {
        "target_col": target_col,
        "n_records": n_records,
        "n_valid_target": n_valid_target,
        "n_missing_target": n_missing_target,
        "missing_target_pct": (
            100.0 * n_missing_target / n_records if n_records > 0 else 0.0
        ),
        "n_classes": distribution.height,
        "majority_class_pct": majority_pct,
        "minority_class_pct": minority_pct,
        "imbalance_ratio": imbalance_ratio,
        "distribution": distribution,
    }

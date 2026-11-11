"""Technical profiling for the Dunnhumby product catalog."""

from __future__ import annotations

import polars as pl

from propensity.eda.profiling import summarize_cardinality
from propensity.eda.table_profile_helpers import categorical_frequencies
from propensity.eda.table_profile_models import ProductProfile

PRODUCT_CATEGORICAL_COLUMNS = (
    "DEPARTMENT",
    "BRAND",
    "COMMODITY_DESC",
    "SUB_COMMODITY_DESC",
    "CURR_SIZE_OF_PRODUCT",
)
PRODUCT_CARDINALITY_COLUMNS = (
    "PRODUCT_ID",
    "MANUFACTURER",
    *PRODUCT_CATEGORICAL_COLUMNS,
)


def _hierarchy_level(
    lf: pl.LazyFrame,
    level: str,
    parent: str | None,
) -> pl.DataFrame:
    aggregations = [pl.col("PRODUCT_ID").n_unique().alias("n_products")]
    if parent is not None:
        aggregations.append(pl.col(parent).n_unique().alias("n_parents"))
    else:
        aggregations.append(pl.lit(1).alias("n_parents"))
    return (
        lf.group_by(level)
        .agg(aggregations)
        .sort("n_products", level, descending=[True, False])
        .collect()
        .select(
            pl.lit(level).alias("level"),
            pl.col(level).cast(pl.String).alias("value"),
            pl.col("n_products").cast(pl.Int64),
            pl.col("n_parents").cast(pl.Int64),
        )
    )


def profile_products(lf: pl.LazyFrame, *, top_n: int = 15) -> ProductProfile:
    """Profile catalog composition without inferring customer demand."""
    overview = lf.select(
        pl.len().alias("n_rows"),
        pl.col("PRODUCT_ID").n_unique().alias("n_products"),
        pl.col("MANUFACTURER").n_unique().alias("n_manufacturers"),
        pl.col("DEPARTMENT").n_unique().alias("n_departments"),
        pl.col("COMMODITY_DESC").n_unique().alias("n_commodities"),
        pl.col("SUB_COMMODITY_DESC").n_unique().alias("n_sub_commodities"),
    ).collect()
    hierarchy = pl.concat(
        (
            _hierarchy_level(lf, "DEPARTMENT", None),
            _hierarchy_level(lf, "COMMODITY_DESC", "DEPARTMENT"),
            _hierarchy_level(lf, "SUB_COMMODITY_DESC", "COMMODITY_DESC"),
        ),
        how="vertical_relaxed",
    )
    return ProductProfile(
        overview=overview,
        cardinality=summarize_cardinality(lf, PRODUCT_CARDINALITY_COLUMNS),
        frequencies=categorical_frequencies(
            lf, PRODUCT_CATEGORICAL_COLUMNS, top_n=top_n
        ),
        hierarchy=hierarchy,
    )


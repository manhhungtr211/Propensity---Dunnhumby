"""Selected cross-feature relationships for transaction technical EDA."""

from __future__ import annotations

import polars as pl

from propensity.eda.transaction_models import (
    RELATIONSHIP_COLUMNS,
    TransactionProfileConfig,
)


def profile_transaction_relationships(
    lf: pl.LazyFrame,
    config: TransactionProfileConfig,
) -> pl.DataFrame:
    """Report selected Pearson relationships at line and household grains."""
    discount_amount = -pl.sum_horizontal(
        [pl.col(column).fill_null(0.0) for column in config.discount_cols]
    )
    line_stats = (
        lf.select(
            pl.col(config.quantity_col).cast(pl.Float64).alias("quantity"),
            pl.col(config.sales_col).cast(pl.Float64).alias("sales"),
            discount_amount.cast(pl.Float64).alias("discount_amount"),
        )
        .select(
            pl.len().alias("n"),
            pl.corr("quantity", "sales").alias("quantity_sales"),
            pl.corr("discount_amount", "sales").alias("discount_sales"),
        )
        .collect()
        .row(0, named=True)
    )
    household_stats = (
        lf.group_by(config.entity_col)
        .agg(
            pl.col(config.basket_col).n_unique().cast(pl.Float64).alias("n_baskets"),
            pl.col(config.sales_col).sum().cast(pl.Float64).alias("total_sales"),
        )
        .select(
            pl.len().alias("n"),
            pl.corr("n_baskets", "total_sales").alias("basket_sales"),
        )
        .collect()
        .row(0, named=True)
    )
    rows = (
        (
            "product_line",
            config.quantity_col,
            config.sales_col,
            line_stats["n"],
            line_stats["quantity_sales"],
        ),
        (
            "product_line",
            "discount_amount",
            config.sales_col,
            line_stats["n"],
            line_stats["discount_sales"],
        ),
        (
            "household",
            "n_baskets",
            "total_sales",
            household_stats["n"],
            household_stats["basket_sales"],
        ),
    )
    return pl.DataFrame(
        rows,
        schema={
            "grain": pl.String,
            "x": pl.String,
            "y": pl.String,
            "n_observations": pl.UInt32,
            "pearson_correlation": pl.Float64,
        },
        orient="row",
    ).select(RELATIONSHIP_COLUMNS)

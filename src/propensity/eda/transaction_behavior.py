"""Analytical basket and household tables for transaction behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from propensity.eda.transaction_models import TransactionProfileConfig

CATEGORY_COLUMN: Final = "DEPARTMENT"
COUPON_COLUMN: Final = "COUPON_DISC"
COUPON_MATCH_COLUMN: Final = "COUPON_MATCH_DISC"
RETAIL_DISCOUNT_COLUMN: Final = "RETAIL_DISC"


@dataclass(frozen=True, slots=True)
class TransactionBehaviorTables:
    """Reusable basket- and household-grain behavioral features."""

    baskets: pl.DataFrame
    households: pl.DataFrame


def _discount_amount(column: str) -> pl.Expr:
    return (-pl.col(column).fill_null(0.0)).clip(lower_bound=0.0)


def _basket_summary(
    transactions: pl.LazyFrame,
    products: pl.LazyFrame,
    config: TransactionProfileConfig,
) -> pl.LazyFrame:
    product_categories = products.select(
        config.product_col,
        pl.col(CATEGORY_COLUMN).alias("category"),
    ).unique(subset=[config.product_col])
    lines = transactions.join(
        product_categories,
        on=config.product_col,
        how="left",
    )
    customer_spend = pl.col(config.sales_col).fill_null(0.0) + pl.col(
        COUPON_COLUMN
    ).fill_null(0.0)
    baskets = lines.group_by(config.basket_col).agg(
        pl.col(config.entity_col).first(),
        pl.col(config.store_col).first(),
        pl.col(config.time_col).first(),
        pl.col(config.period_col).first(),
        pl.col(config.product_col).n_unique().alias("product_count"),
        pl.col("category").drop_nulls().n_unique().alias("category_count"),
        pl.col(config.sales_col).sum().alias("sales_value"),
        pl.col(COUPON_COLUMN).sum().alias("coupon_discount_signed"),
        customer_spend.sum().alias("customer_spend"),
        _discount_amount(RETAIL_DISCOUNT_COLUMN).sum().alias("retail_discount_amount"),
        _discount_amount(COUPON_COLUMN).sum().alias("coupon_discount_amount"),
        _discount_amount(COUPON_MATCH_COLUMN).sum().alias("coupon_match_amount"),
        (
            (pl.col(RETAIL_DISCOUNT_COLUMN) < 0)
            | (pl.col(COUPON_COLUMN) < 0)
            | (pl.col(COUPON_MATCH_COLUMN) < 0)
        )
        .any()
        .alias("has_discount"),
    )
    total_discount = pl.sum_horizontal(
        "retail_discount_amount",
        "coupon_discount_amount",
        "coupon_match_amount",
    )
    return baskets.with_columns(
        (pl.col("customer_spend") > 0).alias("value_is_valid"),
        total_discount.alias("total_discount_amount"),
    ).with_columns(
        pl.when(
            pl.col("value_is_valid")
            & ((pl.col("customer_spend") + pl.col("total_discount_amount")) > 0)
        )
        .then(
            pl.col("total_discount_amount")
            / (pl.col("customer_spend") + pl.col("total_discount_amount"))
        )
        .otherwise(None)
        .alias("discount_rate")
    )


def _household_summary(
    baskets: pl.LazyFrame,
    transactions: pl.LazyFrame,
    products: pl.LazyFrame,
    config: TransactionProfileConfig,
) -> pl.LazyFrame:
    valid_value = pl.col("value_is_valid")
    base = baskets.group_by(config.entity_col).agg(
        pl.len().alias("basket_count"),
        valid_value.sum().alias("valid_value_basket_count"),
        pl.col("customer_spend").filter(valid_value).sum().alias("total_spend"),
        pl.col("customer_spend").filter(valid_value).mean().alias("avg_basket_value"),
        pl.col("customer_spend")
        .filter(valid_value)
        .median()
        .alias("median_basket_value"),
        pl.col("product_count").mean().alias("avg_products_per_basket"),
        pl.col("category_count").mean().alias("avg_categories_per_basket"),
        pl.col(config.store_col).n_unique().alias("unique_stores"),
        pl.col("has_discount").mean().alias("discounted_basket_rate"),
        pl.col("discount_rate").filter(valid_value).mean().alias("avg_discount_rate"),
    )
    primary_store = (
        baskets.group_by(config.entity_col, config.store_col)
        .agg(pl.len().alias("store_baskets"))
        .group_by(config.entity_col)
        .agg(pl.col("store_baskets").max().alias("primary_store_baskets"))
    )
    product_categories = products.select(
        config.product_col,
        pl.col(CATEGORY_COLUMN).alias("category"),
    ).unique(subset=[config.product_col])
    diversity = (
        transactions.select(config.entity_col, config.product_col)
        .unique()
        .join(product_categories, on=config.product_col, how="left")
        .group_by(config.entity_col)
        .agg(
            pl.col(config.product_col).n_unique().alias("unique_products"),
            pl.col("category").drop_nulls().n_unique().alias("category_diversity"),
        )
    )
    return (
        base.join(primary_store, on=config.entity_col, how="left")
        .join(diversity, on=config.entity_col, how="left")
        .with_columns(
            (pl.col("primary_store_baskets") / pl.col("basket_count")).alias(
                "primary_store_share"
            )
        )
        .sort(config.entity_col)
    )


def build_transaction_behavior_tables(
    transactions: pl.LazyFrame,
    products: pl.LazyFrame,
    config: TransactionProfileConfig,
) -> TransactionBehaviorTables:
    """Build the two analytical tables used throughout notebook 03."""
    baskets = _basket_summary(transactions, products, config)
    households = _household_summary(baskets, transactions, products, config)
    return TransactionBehaviorTables(
        baskets=baskets.sort(config.basket_col).collect(),
        households=households.collect(),
    )

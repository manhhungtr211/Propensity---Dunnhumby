"""Deep technical profiling for transaction fact tables."""

from __future__ import annotations

import polars as pl

from propensity.eda._errors import AnalysisConfigurationError
from propensity.eda.transaction_models import (
    GROUP_DISTRIBUTION_COLUMNS,
    IDENTIFIER_FREQUENCY_COLUMNS,
    PERIOD_ACTIVITY_COLUMNS,
    RETURN_INTERVAL_COLUMNS,
    TransactionProfileConfig,
    TransactionTechnicalReport,
)
from propensity.eda.transaction_relationships import profile_transaction_relationships


def _discount_amount(config: TransactionProfileConfig) -> pl.Expr:
    return -pl.sum_horizontal(
        [pl.col(column).fill_null(0.0) for column in config.discount_cols]
    )


def profile_identifier_frequencies(
    lf: pl.LazyFrame,
    columns: tuple[str, ...],
    *,
    top_n: int = 10,
) -> pl.DataFrame:
    """Return the most frequent raw-row values for identifier columns."""
    if top_n <= 0:
        raise AnalysisConfigurationError("top_n phải lớn hơn 0")
    n_rows = int(lf.select(pl.len()).collect().item())
    frames = []
    for column in columns:
        frames.append(
            lf.group_by(column)
            .agg(pl.len().alias("n_rows"))
            .sort("n_rows", column, descending=[True, False])
            .head(top_n)
            .collect()
            .with_row_index("rank", offset=1)
            .select(
                pl.lit(column).alias("column"),
                pl.col(column).cast(pl.String).alias("value"),
                pl.col("n_rows"),
                (pl.col("n_rows") / n_rows).alias("share"),
                pl.col("rank"),
            )
        )
    return pl.concat(frames, how="vertical").select(IDENTIFIER_FREQUENCY_COLUMNS)


def summarize_period_activity(
    lf: pl.LazyFrame,
    period_col: str,
    config: TransactionProfileConfig,
) -> pl.DataFrame:
    """Aggregate transaction activity at one ordinal time grain."""
    return (
        lf.group_by(period_col)
        .agg(
            pl.len().alias("n_product_lines"),
            pl.col(config.basket_col).n_unique().alias("n_baskets"),
            pl.col(config.entity_col).n_unique().alias("n_households"),
            pl.col(config.product_col).n_unique().alias("n_products"),
            pl.col(config.sales_col).sum().alias("total_sales"),
            pl.col(config.quantity_col).sum().alias("total_quantity"),
            _discount_amount(config).sum().alias("total_discount"),
        )
        .sort(period_col)
        .collect()
        .select(
            pl.lit(period_col).alias("period"),
            pl.col(period_col).cast(pl.Int64).alias("period_value"),
            pl.col("n_product_lines"),
            pl.col("n_baskets"),
            pl.col("n_households"),
            pl.col("n_products"),
            pl.col("total_sales").cast(pl.Float64),
            pl.col("total_quantity").cast(pl.Float64),
            pl.col("total_discount").cast(pl.Float64),
        )
        .select(PERIOD_ACTIVITY_COLUMNS)
    )


def _describe_group_metrics(
    grouped: pl.LazyFrame,
    grain: str,
    metrics: tuple[str, ...],
) -> pl.DataFrame:
    expressions: list[pl.Expr] = [pl.len().alias("n_groups")]
    for metric in metrics:
        values = pl.col(metric).cast(pl.Float64)
        expressions.extend(
            (
                values.mean().alias(f"{metric}__mean"),
                values.std().alias(f"{metric}__std"),
                values.quantile(0.25).alias(f"{metric}__p25"),
                values.quantile(0.50).alias(f"{metric}__p50"),
                values.quantile(0.75).alias(f"{metric}__p75"),
                values.quantile(0.90).alias(f"{metric}__p90"),
                values.quantile(0.99).alias(f"{metric}__p99"),
                values.max().alias(f"{metric}__max"),
            )
        )
    statistics = grouped.select(expressions).collect().row(0, named=True)
    rows = [
        (
            grain,
            metric,
            statistics["n_groups"],
            statistics[f"{metric}__mean"],
            statistics[f"{metric}__std"],
            statistics[f"{metric}__p25"],
            statistics[f"{metric}__p50"],
            statistics[f"{metric}__p75"],
            statistics[f"{metric}__p90"],
            statistics[f"{metric}__p99"],
            statistics[f"{metric}__max"],
        )
        for metric in metrics
    ]
    return pl.DataFrame(
        rows,
        schema={
            "grain": pl.String,
            "metric": pl.String,
            "n_groups": pl.UInt32,
            "mean": pl.Float64,
            "std": pl.Float64,
            "p25": pl.Float64,
            "p50": pl.Float64,
            "p75": pl.Float64,
            "p90": pl.Float64,
            "p99": pl.Float64,
            "max": pl.Float64,
        },
        orient="row",
    ).select(GROUP_DISTRIBUTION_COLUMNS)


def summarize_basket_distributions(
    lf: pl.LazyFrame,
    config: TransactionProfileConfig,
) -> pl.DataFrame:
    """Describe product count, sales, quantity and discounts per basket."""
    metrics = (
        "n_product_lines",
        "n_products",
        "basket_sales",
        "basket_quantity",
        "basket_discount",
    )
    grouped = lf.group_by(config.basket_col).agg(
        pl.len().alias("n_product_lines"),
        pl.col(config.product_col).n_unique().alias("n_products"),
        pl.col(config.sales_col).sum().alias("basket_sales"),
        pl.col(config.quantity_col).sum().alias("basket_quantity"),
        _discount_amount(config).sum().alias("basket_discount"),
    )
    return _describe_group_metrics(grouped, "basket", metrics)


def summarize_household_distributions(
    lf: pl.LazyFrame,
    config: TransactionProfileConfig,
) -> pl.DataFrame:
    """Describe repeated purchase activity and value per household."""
    metrics = (
        "n_product_lines",
        "n_baskets",
        "n_active_days",
        "n_products",
        "n_stores",
        "total_sales",
        "total_quantity",
        "total_discount",
        "span_days",
    )
    grouped = lf.group_by(config.entity_col).agg(
        pl.len().alias("n_product_lines"),
        pl.col(config.basket_col).n_unique().alias("n_baskets"),
        pl.col(config.time_col).n_unique().alias("n_active_days"),
        pl.col(config.product_col).n_unique().alias("n_products"),
        pl.col(config.store_col).n_unique().alias("n_stores"),
        pl.col(config.sales_col).sum().alias("total_sales"),
        pl.col(config.quantity_col).sum().alias("total_quantity"),
        _discount_amount(config).sum().alias("total_discount"),
        (
            pl.col(config.time_col).max() - pl.col(config.time_col).min() + 1
        ).alias("span_days"),
    )
    return _describe_group_metrics(grouped, "household", metrics)


def profile_return_intervals(
    lf: pl.LazyFrame,
    config: TransactionProfileConfig,
) -> pl.DataFrame:
    """Summarize gaps between consecutive distinct active days per household."""
    return (
        lf.select(config.entity_col, config.time_col)
        .unique()
        .sort(config.entity_col, config.time_col)
        .with_columns(
            pl.col(config.time_col)
            .diff()
            .over(config.entity_col)
            .alias("return_interval_days")
        )
        .filter(pl.col("return_interval_days").is_not_null())
        .group_by("return_interval_days")
        .agg(pl.len().alias("n_returns"))
        .sort("return_interval_days")
        .with_columns(
            (pl.col("n_returns") / pl.col("n_returns").sum()).alias("share")
        )
        .with_columns(pl.col("share").cum_sum().alias("cumulative_share"))
        .collect()
        .select(RETURN_INTERVAL_COLUMNS)
    )


def build_transaction_technical_report(
    lf: pl.LazyFrame,
    config: TransactionProfileConfig,
    *,
    top_n: int = 10,
) -> TransactionTechnicalReport:
    """Build table-level transaction evidence without business storytelling."""
    return TransactionTechnicalReport(
        identifier_frequencies=profile_identifier_frequencies(
            lf, config.identifier_cols, top_n=top_n
        ),
        daily_activity=summarize_period_activity(lf, config.time_col, config),
        period_activity=summarize_period_activity(lf, config.period_col, config),
        basket_distributions=summarize_basket_distributions(lf, config),
        household_distributions=summarize_household_distributions(lf, config),
        return_intervals=profile_return_intervals(lf, config),
        relationships=profile_transaction_relationships(lf, config),
    )

"""Contracts for table-level technical profiling of transaction data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

IDENTIFIER_FREQUENCY_COLUMNS: Final = (
    "column",
    "value",
    "n_rows",
    "share",
    "rank",
)
PERIOD_ACTIVITY_COLUMNS: Final = (
    "period",
    "period_value",
    "n_product_lines",
    "n_baskets",
    "n_households",
    "n_products",
    "total_sales",
    "total_quantity",
    "total_discount",
)
GROUP_DISTRIBUTION_COLUMNS: Final = (
    "grain",
    "metric",
    "n_groups",
    "mean",
    "std",
    "p25",
    "p50",
    "p75",
    "p90",
    "p99",
    "max",
)
RETURN_INTERVAL_COLUMNS: Final = (
    "return_interval_days",
    "n_returns",
    "share",
    "cumulative_share",
)
RELATIONSHIP_COLUMNS: Final = (
    "grain",
    "x",
    "y",
    "n_observations",
    "pearson_correlation",
)


@dataclass(frozen=True, slots=True)
class TransactionProfileConfig:
    """Column contract required for generic transaction-table profiling."""

    entity_col: str
    basket_col: str
    time_col: str
    period_col: str
    product_col: str
    store_col: str
    quantity_col: str
    sales_col: str
    discount_cols: tuple[str, ...]

    @property
    def identifier_cols(self) -> tuple[str, ...]:
        """Return identifiers whose raw-row frequencies are informative."""
        return (
            self.entity_col,
            self.basket_col,
            self.product_col,
            self.store_col,
        )


@dataclass(frozen=True, slots=True)
class TransactionTechnicalReport:
    """Technical EDA evidence at raw, period, basket and household grains."""

    identifier_frequencies: pl.DataFrame
    daily_activity: pl.DataFrame
    period_activity: pl.DataFrame
    basket_distributions: pl.DataFrame
    household_distributions: pl.DataFrame
    return_intervals: pl.DataFrame
    relationships: pl.DataFrame

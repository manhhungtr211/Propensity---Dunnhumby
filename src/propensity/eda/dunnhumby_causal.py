"""Technical profiling for Dunnhumby product-store-week promotion exposure."""

from __future__ import annotations

import polars as pl

from propensity.eda.table_profile_helpers import (
    categorical_frequencies,
    describe_group_counts,
)
from propensity.eda.table_profile_models import CausalProfile


def profile_causal_data(lf: pl.LazyFrame, *, top_n: int = 15) -> CausalProfile:
    """Profile categorical promotion codes and their structural frequency."""
    overview = lf.select(
        pl.len().alias("n_rows"),
        pl.col("PRODUCT_ID").n_unique().alias("n_products"),
        pl.col("STORE_ID").n_unique().alias("n_stores"),
        pl.col("WEEK_NO").n_unique().alias("n_weeks"),
        pl.col("display").n_unique().alias("n_display_codes"),
        pl.col("mailer").n_unique().alias("n_mailer_codes"),
    ).collect()
    weekly = (
        lf.group_by("WEEK_NO")
        .agg(
            pl.len().alias("n_rows"),
            pl.col("PRODUCT_ID").n_unique().alias("n_products"),
            pl.col("STORE_ID").n_unique().alias("n_stores"),
            pl.col("display").n_unique().alias("n_display_codes"),
            pl.col("mailer").n_unique().alias("n_mailer_codes"),
        )
        .sort("WEEK_NO")
        .collect()
    )
    return CausalProfile(
        overview=overview,
        code_frequencies=categorical_frequencies(
            lf, ("display", "mailer"), top_n=top_n
        ),
        weekly_activity=weekly,
        group_distributions=describe_group_counts(
            lf, ("PRODUCT_ID", "STORE_ID", "WEEK_NO")
        ),
    )


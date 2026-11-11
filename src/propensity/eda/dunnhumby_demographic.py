"""Technical profiling for Dunnhumby household demographics."""

from __future__ import annotations

import polars as pl

from propensity.eda.profiling import summarize_cardinality
from propensity.eda.table_profile_helpers import categorical_frequencies
from propensity.eda.table_profile_models import DemographicProfile

DEMOGRAPHIC_COLUMNS = (
    "AGE_DESC",
    "MARITAL_STATUS_CODE",
    "INCOME_DESC",
    "HOMEOWNER_DESC",
    "HH_COMP_DESC",
    "HOUSEHOLD_SIZE_DESC",
    "KID_CATEGORY_DESC",
)
DEFAULT_SEGMENT_PAIRS = (
    ("INCOME_DESC", "HOUSEHOLD_SIZE_DESC"),
    ("AGE_DESC", "HH_COMP_DESC"),
)


def _segment_counts(
    lf: pl.LazyFrame,
    pairs: tuple[tuple[str, str], ...],
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for left, right in pairs:
        frames.append(
            lf.group_by(left, right)
            .agg(pl.col("household_key").n_unique().alias("n_households"))
            .with_columns(
                (
                    pl.col("n_households") / pl.col("n_households").sum()
                ).alias("share")
            )
            .sort("n_households", descending=True)
            .collect()
            .select(
                pl.lit(f"{left} × {right}").alias("segment"),
                pl.col(left).cast(pl.String).alias("left_value"),
                pl.col(right).cast(pl.String).alias("right_value"),
                pl.col("n_households"),
                pl.col("share"),
            )
        )
    return pl.concat(frames, how="vertical")


def profile_household_demographics(
    lf: pl.LazyFrame,
    *,
    total_households: int,
    top_n: int = 20,
    segment_pairs: tuple[tuple[str, str], ...] = DEFAULT_SEGMENT_PAIRS,
) -> DemographicProfile:
    """Profile demographic coverage, categories and selected segment cross-tabs."""
    available = set(lf.collect_schema().names())
    demographic_columns = tuple(
        column for column in DEMOGRAPHIC_COLUMNS if column in available
    )
    n_households = int(lf.select(pl.col("household_key").n_unique()).collect().item())
    overview = pl.DataFrame(
        {
            "n_households": [n_households],
            "total_households": [total_households],
            "coverage_ratio": [n_households / total_households],
            "n_uncovered_households": [total_households - n_households],
        }
    )
    return DemographicProfile(
        overview=overview,
        cardinality=summarize_cardinality(
            lf, ("household_key", *demographic_columns)
        ),
        frequencies=categorical_frequencies(lf, demographic_columns, top_n=top_n),
        segments=_segment_counts(lf, segment_pairs),
    )

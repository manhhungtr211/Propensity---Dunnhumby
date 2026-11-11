"""Technical profiles for Dunnhumby coupon and campaign tables."""

from __future__ import annotations

import polars as pl

from propensity.eda.table_profile_helpers import (
    categorical_frequencies,
    count_overlapping_intervals,
    describe_group_counts,
    describe_group_metric,
)
from propensity.eda.table_profile_models import (
    CampaignDescriptionProfile,
    CampaignTableProfile,
    CouponMappingProfile,
    RedemptionProfile,
)


def _redemption_return_intervals(lf: pl.LazyFrame) -> pl.DataFrame:
    return (
        lf.select("household_key", "DAY")
        .unique()
        .sort("household_key", "DAY")
        .with_columns(
            pl.col("DAY").diff().over("household_key").alias("interval_days")
        )
        .filter(pl.col("interval_days").is_not_null())
        .group_by("interval_days")
        .agg(pl.len().alias("n_intervals"))
        .sort("interval_days")
        .with_columns(
            (pl.col("n_intervals") / pl.col("n_intervals").sum()).alias("share")
        )
        .collect()
    )


def profile_coupon_redemptions(
    lf: pl.LazyFrame,
    *,
    top_n: int = 15,
) -> RedemptionProfile:
    """Profile coupon redemption events by actor, coupon, campaign and day."""
    overview = lf.select(
        pl.len().alias("n_redemptions"),
        pl.col("household_key").n_unique().alias("n_households"),
        pl.col("COUPON_UPC").n_unique().alias("n_coupons"),
        pl.col("CAMPAIGN").n_unique().alias("n_campaigns"),
        pl.col("DAY").min().alias("min_day"),
        pl.col("DAY").max().alias("max_day"),
    ).collect()
    daily = (
        lf.group_by("DAY")
        .agg(
            pl.len().alias("n_redemptions"),
            pl.col("household_key").n_unique().alias("n_households"),
            pl.col("COUPON_UPC").n_unique().alias("n_coupons"),
        )
        .sort("DAY")
        .collect()
    )
    identifiers = ("household_key", "COUPON_UPC", "CAMPAIGN")
    return RedemptionProfile(
        overview=overview,
        frequencies=categorical_frequencies(lf, identifiers, top_n=top_n),
        group_distributions=describe_group_counts(lf, identifiers),
        daily_activity=daily,
        return_intervals=_redemption_return_intervals(lf),
    )


def profile_campaign_table(
    lf: pl.LazyFrame,
    *,
    total_households: int,
) -> CampaignTableProfile:
    """Profile campaign assignment and household participation."""
    stats = lf.select(
        pl.len().alias("n_rows"),
        pl.col("CAMPAIGN").n_unique().alias("n_campaigns"),
        pl.col("household_key").n_unique().alias("n_households"),
        pl.struct("CAMPAIGN", "household_key").n_unique().alias("n_unique_pairs"),
    ).collect().row(0, named=True)
    n_households = int(stats["n_households"])
    overview = pl.DataFrame(
        {
            **{key: [value] for key, value in stats.items()},
            "duplicate_pairs": [int(stats["n_rows"]) - int(stats["n_unique_pairs"])],
            "total_households": [total_households],
            "coverage_ratio": [n_households / total_households],
        }
    )
    campaign_counts = (
        lf.group_by("CAMPAIGN", "DESCRIPTION")
        .agg(pl.col("household_key").n_unique().alias("n_households"))
        .sort("n_households", descending=True)
        .collect()
    )
    return CampaignTableProfile(
        overview=overview,
        campaign_counts=campaign_counts,
        group_distributions=describe_group_counts(lf, ("household_key", "CAMPAIGN")),
    )


def profile_coupon_mappings(lf: pl.LazyFrame) -> CouponMappingProfile:
    """Profile coupon-product-campaign mapping multiplicity."""
    overview = lf.select(
        pl.len().alias("n_rows"),
        pl.col("COUPON_UPC").n_unique().alias("n_coupons"),
        pl.col("PRODUCT_ID").n_unique().alias("n_products"),
        pl.col("CAMPAIGN").n_unique().alias("n_campaigns"),
        pl.struct("COUPON_UPC", "PRODUCT_ID", "CAMPAIGN")
        .n_unique()
        .alias("n_unique_mappings"),
    ).collect().with_columns(
        (pl.col("n_rows") - pl.col("n_unique_mappings")).alias(
            "n_duplicate_mappings"
        )
    )
    campaign_counts = (
        lf.group_by("CAMPAIGN")
        .agg(
            pl.col("COUPON_UPC").n_unique().alias("n_coupons"),
            pl.col("PRODUCT_ID").n_unique().alias("n_products"),
            pl.len().alias("n_mappings"),
        )
        .sort("CAMPAIGN")
        .collect()
    )
    group_distributions = pl.concat(
        (
            describe_group_metric(
                lf.group_by("COUPON_UPC").agg(
                    pl.col("PRODUCT_ID").n_unique().alias("n_products")
                ),
                grain="COUPON_UPC",
                metric="n_products",
            ),
            describe_group_metric(
                lf.group_by("PRODUCT_ID").agg(
                    pl.col("COUPON_UPC").n_unique().alias("n_coupons")
                ),
                grain="PRODUCT_ID",
                metric="n_coupons",
            ),
            describe_group_metric(
                lf.group_by("CAMPAIGN").agg(
                    pl.col("COUPON_UPC").n_unique().alias("n_coupons")
                ),
                grain="CAMPAIGN",
                metric="n_coupons",
            ),
        ),
        how="vertical",
    )
    return CouponMappingProfile(
        overview=overview,
        group_distributions=group_distributions,
        campaign_counts=campaign_counts,
    )


def profile_campaign_descriptions(lf: pl.LazyFrame) -> CampaignDescriptionProfile:
    """Profile campaign timing, types, durations and temporal overlap."""
    campaigns = (
        lf.with_columns(
            (pl.col("END_DAY") - pl.col("START_DAY") + 1).alias("duration_days")
        )
        .sort("CAMPAIGN")
        .collect()
    )
    overview = pl.DataFrame(
        {
            "n_campaigns": [campaigns.height],
            "min_start_day": [campaigns["START_DAY"].min()],
            "max_end_day": [campaigns["END_DAY"].max()],
            "n_overlapping_pairs": [
                count_overlapping_intervals(
                    campaigns, start_col="START_DAY", end_col="END_DAY"
                )
            ],
        }
    )
    type_counts = (
        campaigns.group_by("DESCRIPTION")
        .agg(pl.len().alias("n_campaigns"))
        .sort("n_campaigns", descending=True)
    )
    duration_summary = campaigns.select(
        pl.col("duration_days").min().alias("min"),
        pl.col("duration_days").mean().alias("mean"),
        pl.col("duration_days").median().alias("median"),
        pl.col("duration_days").max().alias("max"),
    )
    return CampaignDescriptionProfile(
        overview=overview,
        campaigns=campaigns,
        type_counts=type_counts,
        duration_summary=duration_summary,
    )

from __future__ import annotations

import polars as pl

from propensity.eda.dunnhumby_tables import (
    profile_campaign_descriptions,
    profile_campaign_table,
    profile_causal_data,
    profile_coupon_mappings,
    profile_coupon_redemptions,
    profile_household_demographics,
    profile_products,
)


def test_product_profile_covers_catalog_composition_and_hierarchy() -> None:
    frame = pl.LazyFrame(
        {
            "PRODUCT_ID": [1, 2, 3, 4],
            "MANUFACTURER": [10, 10, 11, 12],
            "DEPARTMENT": ["A", "A", "A", "B"],
            "BRAND": ["N", "P", "P", "N"],
            "COMMODITY_DESC": ["X", "X", "Y", "Z"],
            "SUB_COMMODITY_DESC": ["x1", "x2", "y1", "z1"],
            "CURR_SIZE_OF_PRODUCT": ["1 OZ", "2 OZ", "1 OZ", "3 OZ"],
        }
    )

    report = profile_products(frame, top_n=2)

    assert report.overview.row(0, named=True)["n_products"] == 4
    assert set(report.frequencies["column"]) == {
        "DEPARTMENT",
        "BRAND",
        "COMMODITY_DESC",
        "SUB_COMMODITY_DESC",
        "CURR_SIZE_OF_PRODUCT",
    }
    department = report.hierarchy.filter(pl.col("level") == "DEPARTMENT")
    assert department.sort("n_products", descending=True)["n_products"][0] == 3


def test_redemption_profile_counts_events_at_each_relevant_grain() -> None:
    frame = pl.LazyFrame(
        {
            "household_key": [1, 1, 2, 2],
            "DAY": [1, 3, 2, 2],
            "COUPON_UPC": [10, 10, 10, 11],
            "CAMPAIGN": [100, 100, 100, 101],
        }
    )

    report = profile_coupon_redemptions(frame, top_n=2)

    assert report.overview.row(0, named=True)["n_redemptions"] == 4
    assert set(report.group_distributions["grain"]) == {
        "household_key",
        "COUPON_UPC",
        "CAMPAIGN",
    }
    assert report.daily_activity["n_redemptions"].sum() == 4


def test_campaign_and_coupon_profiles_describe_mapping_multiplicity() -> None:
    campaigns = pl.LazyFrame(
        {
            "DESCRIPTION": ["A", "A", "B"],
            "household_key": [1, 2, 1],
            "CAMPAIGN": [10, 10, 11],
        }
    )
    coupons = pl.LazyFrame(
        {
            "COUPON_UPC": [100, 100, 101],
            "PRODUCT_ID": [1, 2, 2],
            "CAMPAIGN": [10, 10, 11],
        }
    )

    campaign_report = profile_campaign_table(campaigns, total_households=4)
    coupon_report = profile_coupon_mappings(coupons)

    assert campaign_report.overview.row(0, named=True)["coverage_ratio"] == 0.5
    assert set(coupon_report.group_distributions["grain"]) == {
        "COUPON_UPC",
        "PRODUCT_ID",
        "CAMPAIGN",
    }


def test_causal_and_demographic_profiles_treat_codes_as_categories() -> None:
    causal = pl.LazyFrame(
        {
            "PRODUCT_ID": [1, 1, 2],
            "STORE_ID": [5, 5, 6],
            "WEEK_NO": [1, 2, 1],
            "display": ["0", "A", "0"],
            "mailer": ["0", "B", "0"],
        }
    )
    demographics = pl.LazyFrame(
        {
            "household_key": [1, 2, 3],
            "AGE_DESC": ["25-34", "25-34", "35-44"],
            "INCOME_DESC": ["A", "B", "A"],
            "HOUSEHOLD_SIZE_DESC": ["1", "2", "2"],
        }
    )

    causal_report = profile_causal_data(causal, top_n=3)
    demographic_report = profile_household_demographics(
        demographics,
        total_households=5,
        segment_pairs=(("INCOME_DESC", "HOUSEHOLD_SIZE_DESC"),),
    )

    assert causal_report.overview.row(0, named=True)["n_rows"] == 3
    assert set(causal_report.code_frequencies["column"]) == {"display", "mailer"}
    assert demographic_report.overview.row(0, named=True)["coverage_ratio"] == 0.6
    assert demographic_report.segments["n_households"].sum() == 3


def test_campaign_description_profile_derives_duration_and_overlap() -> None:
    frame = pl.LazyFrame(
        {
            "DESCRIPTION": ["A", "B", "A"],
            "CAMPAIGN": [1, 2, 3],
            "START_DAY": [1, 5, 20],
            "END_DAY": [10, 15, 25],
        }
    )

    report = profile_campaign_descriptions(frame)

    assert report.campaigns["duration_days"].to_list() == [10, 11, 6]
    assert report.overview.row(0, named=True)["n_overlapping_pairs"] == 1
    assert report.type_counts["n_campaigns"].sum() == 3

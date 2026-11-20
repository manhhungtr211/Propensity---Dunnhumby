"""Unit tests for demographic feature engineering and coverage profiling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propensity.features.demographic import (
    DEMO_GROUPS,
    attribute_excess_table,
    demographic_coverage_profile,
    demographic_features,
    group_demographic,
)


def test_group_demographic_mapping_and_ordering() -> None:
    """Verify group_demographic correctly maps categories into ordered levels."""
    raw = pd.DataFrame(
        {
            "household_key": [1, 2],
            "AGE_DESC": ["65+", "25-34"],
            "MARITAL_STATUS_CODE": ["A", "B"],
            "INCOME_DESC": ["Under 15K", "125-149K"],
            "HOMEOWNER_DESC": ["Homeowner", "Renter"],
            "HH_COMP_DESC": ["Single Female", "2 Adults Kids"],
            "HOUSEHOLD_SIZE_DESC": ["1", "5+"],
            "KID_CATEGORY_DESC": ["None/Unknown", "2"],
        }
    )

    df_grouped = group_demographic(raw)
    assert len(df_grouped) == 2
    assert "demo_age" in df_grouped.columns
    assert "demo_income" in df_grouped.columns

    row1 = df_grouped[df_grouped["household_key"] == 1].iloc[0]
    assert row1["demo_age"] == "55+"
    assert row1["demo_income"] == "<35K"
    assert row1["demo_homeowner"] == "Sở hữu"
    assert row1["demo_marital"] == "A"
    assert row1["demo_hh_comp"] == "Độc thân"
    assert row1["demo_hh_size"] == "1"
    assert row1["demo_kids"] == "Không/không rõ"

    row2 = df_grouped[df_grouped["household_key"] == 2].iloc[0]
    assert row2["demo_age"] == "19-34"
    assert row2["demo_income"] == "125K+"
    assert row2["demo_homeowner"] == "Thuê"
    assert row2["demo_hh_size"] == "4+"
    assert row2["demo_kids"] == "2+"


def test_group_demographic_unmapped_value_raises() -> None:
    """Verify group_demographic raises ValueError when an unmapped value is encountered."""
    raw = pd.DataFrame(
        {
            "household_key": [1],
            "AGE_DESC": ["Alien Age 999"],
            "MARITAL_STATUS_CODE": ["A"],
            "INCOME_DESC": ["Under 15K"],
            "HOMEOWNER_DESC": ["Homeowner"],
            "HH_COMP_DESC": ["Single Female"],
            "HOUSEHOLD_SIZE_DESC": ["1"],
            "KID_CATEGORY_DESC": ["None/Unknown"],
        }
    )
    with pytest.raises(ValueError, match="không map được"):
        group_demographic(raw)


def test_demographic_features_panel_enrichment() -> None:
    """Verify demographic_features merges 0/1 indicator and assigns 'Không có dữ liệu' to missing households."""
    panel = pd.DataFrame(
        {
            "household_key": [1, 2, 3],
            "cutoff_day": [140, 140, 140],
            "target": [1, 0, 1],
            "expected_rf": [0.6, 0.5, 0.4],
        }
    )
    demo_grouped = pd.DataFrame(
        {
            "household_key": [1, 2],
            "demo_age": pd.Categorical(["19-34", "35-44"], categories=["19-34", "35-44", "45-54", "55+"], ordered=True),
        }
    )

    enriched = demographic_features(panel, demo_grouped)
    assert len(enriched) == 3
    assert list(enriched["has_demographic"]) == [1, 1, 0]
    assert enriched.loc[enriched["household_key"] == 3, "demo_age"].iloc[0] == "Không có dữ liệu"


def test_demographic_coverage_profile() -> None:
    """Verify demographic_coverage_profile outputs 2 rows comparing has_demographic 1 vs 0."""
    panel = pd.DataFrame(
        {
            "household_key": [1, 1, 2, 3],
            "cutoff_day": [140, 147, 140, 140],
            "has_demographic": [1, 1, 0, 0],
            "target": [1, 1, 0, 1],
            "expected_rf": [0.7, 0.7, 0.4, 0.4],
            "F_bin": pd.Categorical(["F1", "F1", "F2", "F2"]),
            "recency_bin": pd.Categorical(["R1", "R1", "R2", "R2"]),
        }
    )
    prof = demographic_coverage_profile(panel)
    assert len(prof) == 2
    assert set(prof["has_demographic"]) == {0, 1}


def test_attribute_excess_table() -> None:
    """Verify attribute_excess_table computes excess on has_demographic == 1 subpopulation."""
    # Synthetic panel with 100 rows for 60 households across 2 cutoffs
    records = []
    for c in [140, 147]:
        for hh in range(1, 61):
            records.append(
                {
                    "household_key": hh,
                    "cutoff_day": c,
                    "has_demographic": 1,
                    "target": 1 if hh % 2 == 0 else 0,
                    "expected_rf": 0.50,
                    "demo_age": "19-34" if hh <= 30 else "35-44",
                }
            )
    panel = pd.DataFrame(records)
    panel["demo_age"] = pd.Categorical(panel["demo_age"], categories=["19-34", "35-44"])

    tbl = attribute_excess_table(panel, attrs=["age"])
    assert len(tbl) == 2
    assert "attribute" in tbl.columns
    assert "level" in tbl.columns
    assert "excess_pp" in tbl.columns
    assert "verdict" in tbl.columns

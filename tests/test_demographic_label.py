from __future__ import annotations

import pandas as pd
import pytest

from propensity.eda.demographic_label import (
    FEATURES,
    build_demographic_label_report,
    prepare_demographic_label_panel,
)


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "household_key": [1, 1, 2, 2, 3, 3],
            "cutoff_day": [10, 20, 10, 20, 10, 20],
            "label": [0, 1, 0, 1, 0, 0],
        }
    )


def _demographics() -> pd.DataFrame:
    values = {
        "AGE_DESC": ["25-34", "45-54", "65+"],
        "MARITAL_STATUS_CODE": ["A", "B", "U"],
        "INCOME_DESC": ["25-34K", "50-74K", "100-124K"],
        "HOMEOWNER_DESC": ["Renter", "Homeowner", "Unknown"],
        "HH_COMP_DESC": ["Single Female", "2 Adults No Kids", "Unknown"],
        "HOUSEHOLD_SIZE_DESC": ["1", "2", "3"],
        "KID_CATEGORY_DESC": ["None/Unknown", "None/Unknown", "1"],
    }
    return pd.DataFrame({"household_key": [1, 2, 3], **values})


def test_distribution_shares_sum_to_one_within_each_label() -> None:
    # Given
    panel = prepare_demographic_label_panel(_labels(), _demographics())

    # When
    report = build_demographic_label_report(panel)

    # Then
    sums = report.distribution_summary.groupby(["feature", "label"])[
        "within_label_share"
    ].sum()
    assert len(sums) == len(FEATURES) * 2
    assert sums.to_numpy() == pytest.approx(1.0)


def test_overlap_counts_households_observed_in_both_labels() -> None:
    # Given
    panel = prepare_demographic_label_panel(_labels(), _demographics())

    # When
    overlap = build_demographic_label_report(panel).household_label_overlap.iloc[0]

    # Then
    assert overlap["n_households"] == 3
    assert overlap["n_households_in_both_labels"] == 2
    assert overlap["share_households_in_both_labels"] == pytest.approx(2 / 3)


def test_positive_rate_is_calculated_within_category() -> None:
    # Given
    panel = prepare_demographic_label_panel(_labels(), _demographics())

    # When
    summary = build_demographic_label_report(panel).positive_rate_summary
    age_65 = summary.loc[
        (summary["feature"] == "AGE_DESC") & (summary["category"] == "65+")
    ].iloc[0]

    # Then
    assert age_65["n_observations"] == 2
    assert age_65["n_positive"] == 0
    assert age_65["n_negative"] == 2
    assert age_65["positive_rate"] == 0


def test_distribution_comparison_is_conditioned_on_label() -> None:
    # Given
    panel = prepare_demographic_label_panel(_labels(), _demographics())

    # When
    comparison = build_demographic_label_report(panel).category_gap_summary
    age_65 = comparison.loc[
        (comparison["feature"] == "AGE_DESC")
        & (comparison["category"] == "65+")
    ].iloc[0]

    # Then
    assert age_65["Positive"] == 0
    assert age_65["Negative"] == pytest.approx(0.5)
    assert age_65["positive_minus_negative_pp"] == pytest.approx(-50.0)


def test_missing_demographic_group_is_excluded_from_distance() -> None:
    # Given
    labels = pd.concat(
        [
            _labels(),
            pd.DataFrame(
                {
                    "household_key": [4, 4],
                    "cutoff_day": [10, 20],
                    "label": [0, 0],
                }
            ),
        ],
        ignore_index=True,
    )
    panel = prepare_demographic_label_panel(labels, _demographics())

    # When
    report = build_demographic_label_report(panel)

    # Then
    assert "NO_DEMOGRAPHIC_RECORD" in set(
        report.distribution_summary["category"]
    )
    assert "NO_DEMOGRAPHIC_RECORD" not in set(report.category_gap_summary["category"])

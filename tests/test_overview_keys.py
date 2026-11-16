"""Behavioral tests for identifier and candidate grain profiling."""

import pandas as pd
import pytest

from propensity.overview.keys import profile_candidate_grain, profile_identifiers
from propensity.overview.models import (
    GRAIN_SUMMARY_CONTRACT,
    IDENTIFIER_SUMMARY_CONTRACT,
)
from propensity.schema.spec import DatasetSpec


def test_profile_identifiers_unique_and_invalid_when_missing() -> None:
    # Given:
    # id1 has distinct values [1, 2, 3] -> unique and valid
    # id2 has distinct values [1, 2, None] -> unique among non-missing, but invalid because missing > 0
    # id3 has duplicates [1, 1, 2] -> not unique, not valid
    frame = pd.DataFrame(
        {
            "id1": [1, 2, 3],
            "id2": [1, 2, None],
            "id3": [1, 1, 2],
        }
    )
    spec = DatasetSpec(
        name="test_id",
        identifier_cols=("id1", "id2", "id3"),
    )

    # When
    result = profile_identifiers(frame, table_name="test_id", spec=spec)

    # Then
    assert tuple(result.columns) == IDENTIFIER_SUMMARY_CONTRACT.columns
    assert len(result) == 3

    r1 = result[result["column"] == "id1"].iloc[0]
    assert r1["n_rows"] == 3
    assert r1["n_missing"] == 0
    assert r1["n_unique_non_missing"] == 3
    assert bool(r1["is_unique"]) is True
    assert bool(r1["is_valid"]) is True

    r2 = result[result["column"] == "id2"].iloc[0]
    assert r2["n_missing"] == 1
    assert r2["missing_pct"] == pytest.approx(33.333, rel=1e-2)
    assert r2["n_unique_non_missing"] == 2
    assert bool(r2["is_unique"]) is True
    assert bool(r2["is_valid"]) is False  # Missing keys invalidate the identifier

    r3 = result[result["column"] == "id3"].iloc[0]
    assert r3["n_missing"] == 0
    assert r3["n_unique_non_missing"] == 2
    assert bool(r3["is_unique"]) is False
    assert bool(r3["is_valid"]) is False


def test_profile_candidate_grain_single_and_composite() -> None:
    # Given:
    # (A, B) grain:
    # (1, 'x'), (1, 'y'), (2, 'x'), (2, 'x'), (3, None)
    # Total rows = 5
    # Complete key rows = 4 ((1, 'x'), (1, 'y'), (2, 'x'), (2, 'x'))
    # Missing key rows = 1
    # Among complete keys:
    # Groups: (1, 'x') [1], (1, 'y') [1], (2, 'x') [2]
    # unique_groups = 2, duplicate_groups = 1, excess_rows = 1, max_rows = 2
    # is_unique = False, is_valid = False
    frame = pd.DataFrame(
        {
            "A": [1, 1, 2, 2, 3],
            "B": ["x", "y", "x", "x", None],
        }
    )
    spec = DatasetSpec(
        name="composite",
        grain_cols=("A", "B"),
    )

    # When
    result = profile_candidate_grain(frame, table_name="composite", spec=spec)

    # Then
    assert tuple(result.columns) == GRAIN_SUMMARY_CONTRACT.columns
    assert len(result) == 1
    row = result.iloc[0]
    assert row["grain"] == "A, B"
    assert row["n_rows"] == 5
    assert row["complete_key_rows"] == 4
    assert row["missing_key_rows"] == 1
    assert row["unique_groups"] == 2
    assert row["duplicate_groups"] == 1
    assert row["excess_rows"] == 1
    assert row["max_rows_per_group"] == 2
    assert bool(row["is_unique"]) is False
    assert bool(row["is_valid"]) is False


def test_profile_candidate_grain_unique_with_missing_key() -> None:
    # Given:
    # Complete keys are (1, 'x'), (2, 'y') -> unique
    # Row 3 is (3, None) -> missing key
    frame = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": ["x", "y", "   "],
        }
    )
    spec = DatasetSpec(
        name="test_grain",
        grain_cols=("A", "B"),
    )

    # When
    result = profile_candidate_grain(frame, table_name="test_grain", spec=spec)

    # Then
    row = result.iloc[0]
    assert row["complete_key_rows"] == 2
    assert row["missing_key_rows"] == 1
    assert row["unique_groups"] == 2
    assert row["duplicate_groups"] == 0
    assert row["excess_rows"] == 0
    assert bool(row["is_unique"]) is True
    assert bool(row["is_valid"]) is False


def test_profile_candidate_grain_empty_frame() -> None:
    # Given
    frame = pd.DataFrame(columns=["A", "B"])
    spec = DatasetSpec(name="empty", grain_cols=("A", "B"))

    # When
    result = profile_candidate_grain(frame, table_name="empty", spec=spec)

    # Then
    assert len(result) == 1
    row = result.iloc[0]
    assert row["n_rows"] == 0
    assert row["complete_key_rows"] == 0
    assert row["missing_key_rows"] == 0
    assert bool(row["is_unique"]) is True

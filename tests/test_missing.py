"""Missingness treats blank strings consistently across pandas dtypes."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from propensity.overview.missing import blank_mask, missing_mask


@pytest.mark.parametrize("dtype", ["object", "string", "category"])
def test_missing_mask_when_strings_include_nulls_and_blanks(dtype: str) -> None:
    # Given
    series = pd.Series(
        [None, pd.NA, float("nan"), "", " \t\n", "\u2003", "0", " 01 ", "na"],
        dtype=dtype,
        index=list("abcdefghi"),
        name="code",
    )
    expected = pd.Series(
        [True, True, True, True, True, True, False, False, False],
        index=series.index,
        name="code",
    )
    # When
    result = missing_mask(series)
    # Then
    assert_series_equal(result, expected)


def test_blank_mask_when_nulls_are_distinguished_from_blanks() -> None:
    # Given
    series = pd.Series([None, "", " \t", "0", 0, 1.5, False])
    # When
    result = blank_mask(series)
    # Then
    assert_series_equal(
        result, pd.Series([False, True, True, False, False, False, False])
    )


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([0, pd.NA, 2], dtype="Int64"),
        pd.Series([0.0, float("nan"), 2.5]),
        pd.Series([False, pd.NA, True], dtype="boolean"),
        pd.Series([pd.Timestamp("2024-01-01"), pd.NaT, pd.Timestamp("2024-01-03")]),
        pd.Series([0, None, 2], dtype="object"),
    ],
)
def test_missing_mask_when_nonstring_values_include_nulls(series: pd.Series) -> None:
    # Given
    expected = pd.Series([False, True, False])
    # When
    result = missing_mask(series)
    # Then
    assert_series_equal(result, expected)


def test_missing_mask_when_empty_returns_empty_boolean_series() -> None:
    # Given
    series = pd.Series([], dtype="string", name="empty")
    # When
    result = missing_mask(series)
    # Then
    assert_series_equal(result, pd.Series([], dtype=bool, name="empty"))


def test_missing_mask_when_frame_column_is_passed_preserves_source() -> None:
    # Given
    frame = pd.DataFrame({"code": [None, " \t", " 01 ", "0", 0]})
    original = frame.copy(deep=True)
    # When
    missing_mask(frame["code"])
    # Then
    assert_frame_equal(frame, original)

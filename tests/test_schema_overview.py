"""Behavioral coverage for explicit dataset semantics."""

from dataclasses import replace

import pandas as pd
import pytest

from propensity.schema import ColumnRole, DatasetSpec, DatasetSpecError, resolve_schema


def test_explicit_time_roles_when_ordinal_clock_and_calendar_coexist() -> None:
    # Given
    frame = pd.DataFrame(
        {
            "DAY": [1],
            "WEEK_NO": [2],
            "TRANS_TIME": [930],
            "date": [pd.Timestamp("2024-01-01")],
        }
    )
    spec = DatasetSpec(
        name="time",
        ordinal_time_cols=("DAY", "WEEK_NO"),
        clock_time_cols=("TRANS_TIME",),
    )
    # When
    roles = resolve_schema(frame, spec)
    # Then
    assert roles == {
        "DAY": ColumnRole.TIME,
        "WEEK_NO": ColumnRole.TIME,
        "TRANS_TIME": ColumnRole.TIME,
        "date": ColumnRole.DATETIME,
    }


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (
            DatasetSpec("x", entity_col="x", target_col="x", identifier_cols=("x",)),
            ColumnRole.ENTITY,
        ),
        (
            DatasetSpec("x", target_col="x", identifier_cols=("x",), time_col="x"),
            ColumnRole.TARGET,
        ),
        (DatasetSpec("x", identifier_cols=("x",), datetime_cols=("x",)), ColumnRole.ID),
        (DatasetSpec("x", time_col="x", datetime_cols=("x",)), ColumnRole.DATETIME),
        (DatasetSpec("x", time_col="x"), ColumnRole.TIME),
        (DatasetSpec("x", interval_cols=(("x", "end"),)), ColumnRole.TIME),
        (DatasetSpec("x"), ColumnRole.NUMERIC),
    ],
)
def test_semantic_precedence_when_declarations_overlap(
    spec: DatasetSpec, expected: ColumnRole
) -> None:
    # Given
    frame = pd.DataFrame({"x": [1]})
    # When
    roles = resolve_schema(frame, spec)
    # Then
    assert roles["x"] == expected


def test_calendar_role_when_time_column_has_datetime_dtype() -> None:
    # Given
    frame = pd.DataFrame({"x": [pd.Timestamp("2024-01-01")]})
    spec = DatasetSpec("dates", time_col="x")
    # When
    roles = resolve_schema(frame, spec)
    # Then
    assert roles["x"] == ColumnRole.DATETIME


def test_numeric_role_when_time_like_names_have_no_metadata() -> None:
    # Given
    frame = pd.DataFrame({"DAY": [1], "WEEK_NO": [2], "TRANS_TIME": [930]})
    # When
    roles = resolve_schema(frame, DatasetSpec("raw"))
    # Then
    assert all(role == ColumnRole.NUMERIC for role in roles.values())


@pytest.mark.parametrize(
    "field",
    [
        "grain_cols",
        "identifier_cols",
        "datetime_cols",
        "required_cols",
        "string_cols",
        "ordinal_time_cols",
        "clock_time_cols",
    ],
)
@pytest.mark.parametrize("columns", [(" ",), ("x", "x")])
def test_rejects_invalid_column_lists_when_blank_or_duplicate(
    field: str, columns: tuple[str, ...]
) -> None:
    # Given
    spec = DatasetSpec("raw")
    # When / Then
    with pytest.raises(DatasetSpecError):
        replace(spec, **{field: columns})


@pytest.mark.parametrize(
    "field", ["name", "entity_col", "time_col", "target_col", "time_grain"]
)
def test_rejects_blank_scalar_metadata_when_whitespace_only(field: str) -> None:
    # Given
    spec = DatasetSpec("raw", time_col="time")
    # When / Then
    with pytest.raises(DatasetSpecError):
        replace(spec, **{field: " "})


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("datetime_cols", "ordinal_time_cols"),
        ("datetime_cols", "clock_time_cols"),
        ("ordinal_time_cols", "clock_time_cols"),
        ("string_cols", "datetime_cols"),
    ],
)
def test_rejects_overlap_when_encodings_conflict(first: str, second: str) -> None:
    # Given
    spec = DatasetSpec("raw")
    # When / Then
    with pytest.raises(DatasetSpecError):
        replace(spec, **{first: ("x",), second: ("x",)})


@pytest.mark.parametrize(
    "pair",
    [("start",), ("start", "end", "extra"), ("", "end"), ("start", " "), ("x", "x")],
)
def test_rejects_interval_when_endpoints_are_malformed(pair: tuple[str, ...]) -> None:
    # Given
    spec = DatasetSpec("interval")
    # When / Then
    with pytest.raises(DatasetSpecError):
        replace(spec, interval_cols=(pair,))


def test_rejects_interval_when_pair_is_duplicated() -> None:
    # Given
    pair = ("start", "end")
    # When / Then
    with pytest.raises(DatasetSpecError):
        DatasetSpec("interval", interval_cols=(pair, pair))


@pytest.mark.parametrize("encoding", ["ordinal_time_cols", "datetime_cols"])
def test_accepts_interval_when_endpoint_encoding_is_explicit(encoding: str) -> None:
    # Given
    spec = DatasetSpec("interval", interval_cols=(("start", "end"),))
    frame = pd.DataFrame({"start": [1], "end": [2]})
    # When
    encoded = replace(spec, **{encoding: ("start", "end")})
    encoded.validate(frame)
    # Then
    assert encoded.required_columns == ("start", "end")


def test_requires_interval_end_when_column_is_missing() -> None:
    # Given
    spec = DatasetSpec("interval", interval_cols=(("start", "end"),))
    frame = pd.DataFrame({"start": [1]})
    # When / Then
    with pytest.raises(DatasetSpecError):
        spec.validate(frame)


def test_accepts_structural_overlap_when_identifier_is_string_encoded() -> None:
    # Given
    frame = pd.DataFrame({"code": ["001"]})
    # When
    spec = DatasetSpec(
        "codes",
        entity_col="code",
        grain_cols=("code",),
        identifier_cols=("code",),
        required_cols=("code",),
        string_cols=("code",),
    )
    spec.validate(frame)
    # Then
    assert spec.required_columns == ("code",)


def test_target_is_optional_when_absent_from_frame() -> None:
    # Given
    frame = pd.DataFrame({"code": [1]})
    spec = DatasetSpec("raw", identifier_cols=("code",), target_col="target")
    # When
    spec.validate(frame)
    # Then
    assert spec.required_columns == ("code",)

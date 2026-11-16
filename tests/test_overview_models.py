"""Behavioral contract for the seven overview result tables."""

from dataclasses import FrozenInstanceError, replace
from typing import Final

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from propensity.overview.models import (
    RESULT_CONTRACTS,
    OverviewContractError,
    OverviewReport,
    ResultContract,
    empty_result_frame,
)

EXPECTED: Final = (
    (
        "table_summary",
        "table n_rows n_columns memory_bytes",
        "string Int64 Int64 Int64",
    ),
    (
        "column_summary",
        "table column dtype role is_required is_identifier is_grain is_time",
        "string string string string boolean boolean boolean boolean",
    ),
    (
        "identifier_summary",
        "table column n_rows n_missing missing_pct n_unique_non_missing is_unique is_valid",
        "string string Int64 Int64 Float64 Int64 boolean boolean",
    ),
    (
        "grain_summary",
        (
            "table grain n_rows complete_key_rows missing_key_rows unique_groups "
            "duplicate_groups excess_rows max_rows_per_group is_unique is_valid"
        ),
        "string string Int64 Int64 Int64 Int64 Int64 Int64 Int64 boolean boolean",
    ),
    (
        "time_summary",
        (
            "table column kind unit start_column end_column min max n_unique_non_missing "
            "span n_missing missing_endpoint_rows start_after_end_rows invalid_clock_rows"
        ),
        (
            "string string string string string string object object Int64 object "
            "Int64 Int64 Int64 Int64"
        ),
    ),
    (
        "quality_summary",
        (
            "table column n_rows n_missing n_blank missing_pct n_unique_non_missing "
            "is_all_missing is_constant duplicate_groups excess_rows duplicate_pct"
        ),
        "string string Int64 Int64 Int64 Float64 Int64 boolean boolean Int64 Int64 Float64",
    ),
    (
        "numeric_summary",
        "table column count mean std min p25 median p75 max",
        "string string Int64 Float64 Float64 Float64 Float64 Float64 Float64 Float64",
    ),
    (
        "provenance",
        (
            "table path size_bytes scan_mode chunk_rows exact rows_scanned "
            "spill_partitions max_chunk_rows max_partition_rows"
        ),
        "string string Int64 string Int64 boolean Int64 Int64 Int64 Int64",
    ),
)


@pytest.mark.parametrize("name,columns,dtypes", EXPECTED)
def test_empty_report_preserves_contract_when_no_rows(
    name: str, columns: str, dtypes: str
) -> None:
    # Given: a result schema whose order and nullable types are public API.
    expected_columns = tuple(columns.split())
    expected_dtypes = dtypes.split()
    # When: constructing a default empty report.
    result = getattr(OverviewReport(), name)
    # Then: it is a plain empty pandas frame with the specified schema.
    assert type(result) is pd.DataFrame
    assert result.empty
    assert tuple(result.columns) == expected_columns
    assert result.dtypes.astype(str).tolist() == expected_dtypes


@pytest.mark.parametrize("contract", RESULT_CONTRACTS, ids=lambda value: value.name)
def test_populated_report_normalizes_when_columns_are_reversed(
    contract: ResultContract,
) -> None:
    # Given: populated nullable columns supplied out of order.
    values = {
        "string": "source",
        "Int64": 3,
        "Float64": 12.5,
        "boolean": True,
        "object": pd.Timestamp("2024-01-01"),
    }
    source = pd.DataFrame(
        {name: [values[dtype], None] for name, dtype in reversed(contract.fields)}
    )
    before = source.copy(deep=True)
    # When: grouping the frame into a report.
    report = replace(OverviewReport(), **{contract.name: source})
    result = getattr(report, contract.name)
    # Then: the contract is normalized without modifying the input.
    assert tuple(result.columns) == contract.columns
    assert result.dtypes.astype(str).tolist() == [dtype for _, dtype in contract.fields]
    assert result.iloc[1].isna().all()
    for name, dtype in contract.fields:
        assert result.loc[0, name] == values[dtype]
    assert_frame_equal(source, before)
    result.loc[0, "table"] = "changed"
    assert source.loc[0, "table"] == "source"


@pytest.mark.parametrize("contract", RESULT_CONTRACTS, ids=lambda value: value.name)
def test_report_raises_typed_error_when_required_columns_are_missing(
    contract: ResultContract,
) -> None:
    # Given: a result frame with two required columns absent.
    missing = contract.columns[:2]
    malformed = empty_result_frame(contract).drop(columns=list(missing))
    # When: grouping the malformed frame into a report.
    with pytest.raises(OverviewContractError) as raised:
        replace(OverviewReport(), **{contract.name: malformed})
    # Then: callers can identify the result and each missing column.
    assert raised.value.result_name == contract.name
    assert raised.value.missing_columns == missing
    assert str(raised.value)


def test_report_fields_are_frozen_when_reassigned() -> None:
    # Given: a valid empty report.
    report = OverviewReport()
    # When: attempting to replace a report field.
    with pytest.raises(FrozenInstanceError):
        report.table_summary = pd.DataFrame()
    # Then: the report has slots rather than an open attribute dictionary.
    assert not hasattr(report, "__dict__")


def test_empty_frames_are_independent_when_one_is_changed() -> None:
    # Given: two independently constructed reports.
    first, second = OverviewReport(), OverviewReport()
    # When: adding a row to the first pandas frame.
    first.table_summary.loc[0] = ["one", 1, 1, 1]
    # Then: the other report keeps its own empty result.
    assert second.table_summary.empty


def test_time_values_retain_semantics_when_report_is_constructed() -> None:
    # Given: native ordinal, datetime and interval span values.
    source = pd.concat([OverviewReport().time_summary] * 3, ignore_index=True)
    source.loc[0, ["kind", "min", "max", "span"]] = ["ordinal", 1, 711, 711]
    source.loc[1, ["kind", "min", "max", "span"]] = [
        "datetime",
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-03"),
        pd.Timedelta(days=2),
    ]
    source.loc[2, ["kind", "min", "max", "span"]] = ["clock", 0, 2359, pd.NA]
    # When: grouping the time frame into a report.
    result = OverviewReport(time_summary=source).time_summary
    # Then: ordinal/clock numbers remain numbers and datetime retains its type.
    assert result.loc[0, "min"] == 1
    assert result.loc[0, "max"] == 711
    assert result.loc[1, "min"] == pd.Timestamp("2024-01-01")
    assert result.loc[1, "span"] == pd.Timedelta(days=2)
    assert result.loc[2, "max"] == 2359
    assert pd.isna(result.loc[2, "span"])

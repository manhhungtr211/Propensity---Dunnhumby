"""Behavioral tests for semantic time coverage profiling."""

import pandas as pd

from propensity.overview.models import TIME_SUMMARY_CONTRACT
from propensity.overview.temporal import profile_time_coverage
from propensity.schema.spec import DatasetSpec


def test_profile_time_coverage_all_time_kinds() -> None:
    # Given:
    # ordinal DAY: 1, 5, 10
    # clock TRANS_TIME: 930 (09:30), 2359 (23:59), 2360 (invalid minute 60!)
    # interval START_DAY, END_DAY: (10, 20), (25, 20) [reversed!], (30, None) [missing endpoint]
    frame = pd.DataFrame(
        {
            "DAY": [1, 5, 10],
            "TRANS_TIME": [930, 2359, 2360],
            "START_DAY": [10, 25, 30],
            "END_DAY": [20, 20, None],
            "ts": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05"), None],
        }
    )
    spec = DatasetSpec(
        name="time_test",
        time_col="DAY",
        time_grain="day_index",
        ordinal_time_cols=("DAY",),
        clock_time_cols=("TRANS_TIME",),
        interval_cols=(("START_DAY", "END_DAY"),),
        datetime_cols=("ts",),
    )

    # When
    result = profile_time_coverage(frame, table_name="time_test", spec=spec)

    # Then
    assert tuple(result.columns) == TIME_SUMMARY_CONTRACT.columns
    assert len(result) == 4

    day_row = result[result["column"] == "DAY"].iloc[0]
    assert day_row["kind"] == "ordinal"
    assert day_row["min"] == 1
    assert day_row["max"] == 10
    assert day_row["span"] == 9
    assert day_row["n_missing"] == 0

    clock_row = result[result["column"] == "TRANS_TIME"].iloc[0]
    assert clock_row["kind"] == "clock"
    assert clock_row["unit"] == "HHMM"
    assert pd.isna(clock_row["span"])  # Clock codes must not be aggregated as duration
    assert clock_row["invalid_clock_rows"] == 1  # 2360 is invalid

    interval_row = result[result["kind"] == "interval"].iloc[0]
    assert interval_row["start_column"] == "START_DAY"
    assert interval_row["end_column"] == "END_DAY"
    assert interval_row["min"] == 10
    assert interval_row["max"] == 20
    assert interval_row["missing_endpoint_rows"] == 1  # Row 2 has END_DAY None
    assert interval_row["start_after_end_rows"] == 1  # 25 > 20 is reversed

    ts_row = result[result["column"] == "ts"].iloc[0]
    assert ts_row["kind"] == "datetime"
    assert ts_row["n_missing"] == 1
    assert ts_row["n_unique_non_missing"] == 2


def test_profile_time_coverage_when_no_time_configured() -> None:
    # Given
    frame = pd.DataFrame({"x": [1, 2]})
    spec = DatasetSpec(name="no_time")

    # When
    result = profile_time_coverage(frame, table_name="no_time", spec=spec)

    # Then
    assert tuple(result.columns) == TIME_SUMMARY_CONTRACT.columns
    assert len(result) == 0

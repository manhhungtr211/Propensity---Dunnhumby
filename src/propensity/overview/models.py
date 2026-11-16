"""Stable, unstyled pandas result contracts for source-table overviews."""

from dataclasses import dataclass, field
from functools import partial
from typing import Final

import pandas as pd


class OverviewContractError(ValueError):
    """A result frame lacks columns required by its named contract."""

    __slots__ = ("missing_columns", "result_name")

    def __init__(self, result_name: str, missing_columns: tuple[str, ...]) -> None:
        self.result_name = result_name
        self.missing_columns = missing_columns
        super().__init__(f"{result_name}: missing result columns {missing_columns}")


@dataclass(frozen=True, slots=True)
class ResultContract:
    """Ordered result columns and their pandas storage types."""

    name: str
    fields: tuple[tuple[str, str], ...]

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.fields)


TABLE_SUMMARY_CONTRACT: Final = ResultContract(
    "table_summary",
    (
        ("table", "string"),
        ("n_rows", "Int64"),
        ("n_columns", "Int64"),
        ("memory_bytes", "Int64"),
    ),
)
COLUMN_SUMMARY_CONTRACT: Final = ResultContract(
    "column_summary",
    (
        ("table", "string"),
        ("column", "string"),
        ("dtype", "string"),
        ("role", "string"),
        ("is_required", "boolean"),
        ("is_identifier", "boolean"),
        ("is_grain", "boolean"),
        ("is_time", "boolean"),
    ),
)
IDENTIFIER_SUMMARY_CONTRACT: Final = ResultContract(
    "identifier_summary",
    (
        ("table", "string"),
        ("column", "string"),
        ("n_rows", "Int64"),
        ("n_missing", "Int64"),
        ("missing_pct", "Float64"),
        ("n_unique_non_missing", "Int64"),
        ("is_unique", "boolean"),
        ("is_valid", "boolean"),
    ),
)
GRAIN_SUMMARY_CONTRACT: Final = ResultContract(
    "grain_summary",
    (
        ("table", "string"),
        ("grain", "string"),
        ("n_rows", "Int64"),
        ("complete_key_rows", "Int64"),
        ("missing_key_rows", "Int64"),
        ("unique_groups", "Int64"),
        ("duplicate_groups", "Int64"),
        ("excess_rows", "Int64"),
        ("max_rows_per_group", "Int64"),
        ("is_unique", "boolean"),
        ("is_valid", "boolean"),
    ),
)
TIME_SUMMARY_CONTRACT: Final = ResultContract(
    "time_summary",
    (
        ("table", "string"),
        ("column", "string"),
        ("kind", "string"),
        ("unit", "string"),
        ("start_column", "string"),
        ("end_column", "string"),
        ("min", "object"),
        ("max", "object"),
        ("n_unique_non_missing", "Int64"),
        ("span", "object"),
        ("n_missing", "Int64"),
        ("missing_endpoint_rows", "Int64"),
        ("start_after_end_rows", "Int64"),
        ("invalid_clock_rows", "Int64"),
    ),
)
QUALITY_SUMMARY_CONTRACT: Final = ResultContract(
    "quality_summary",
    (
        ("table", "string"),
        ("column", "string"),
        ("n_rows", "Int64"),
        ("n_missing", "Int64"),
        ("n_blank", "Int64"),
        ("missing_pct", "Float64"),
        ("n_unique_non_missing", "Int64"),
        ("is_all_missing", "boolean"),
        ("is_constant", "boolean"),
        ("duplicate_groups", "Int64"),
        ("excess_rows", "Int64"),
        ("duplicate_pct", "Float64"),
    ),
)
PROVENANCE_CONTRACT: Final = ResultContract(
    "provenance",
    (
        ("table", "string"),
        ("path", "string"),
        ("size_bytes", "Int64"),
        ("scan_mode", "string"),
        ("chunk_rows", "Int64"),
        ("exact", "boolean"),
        ("rows_scanned", "Int64"),
        ("spill_partitions", "Int64"),
        ("max_chunk_rows", "Int64"),
        ("max_partition_rows", "Int64"),
    ),
)

TABLE_SUMMARY_COLUMNS: Final = TABLE_SUMMARY_CONTRACT.columns
COLUMN_SUMMARY_COLUMNS: Final = COLUMN_SUMMARY_CONTRACT.columns
IDENTIFIER_SUMMARY_COLUMNS: Final = IDENTIFIER_SUMMARY_CONTRACT.columns
GRAIN_SUMMARY_COLUMNS: Final = GRAIN_SUMMARY_CONTRACT.columns
TIME_SUMMARY_COLUMNS: Final = TIME_SUMMARY_CONTRACT.columns
QUALITY_SUMMARY_COLUMNS: Final = QUALITY_SUMMARY_CONTRACT.columns
PROVENANCE_COLUMNS: Final = PROVENANCE_CONTRACT.columns
RESULT_CONTRACTS: Final = (
    TABLE_SUMMARY_CONTRACT,
    COLUMN_SUMMARY_CONTRACT,
    IDENTIFIER_SUMMARY_CONTRACT,
    GRAIN_SUMMARY_CONTRACT,
    TIME_SUMMARY_CONTRACT,
    QUALITY_SUMMARY_CONTRACT,
    PROVENANCE_CONTRACT,
)


def empty_result_frame(contract: ResultContract) -> pd.DataFrame:
    """Create an empty frame with ordered columns and nullable metric dtypes."""
    return pd.DataFrame(
        {name: pd.Series(dtype=dtype) for name, dtype in contract.fields}
    )


def normalize_result_frame(
    frame: pd.DataFrame, contract: ResultContract
) -> pd.DataFrame:
    """Copy required columns in contract order, preserving nullable metric types."""
    missing = tuple(name for name in contract.columns if name not in frame.columns)
    if missing:
        raise OverviewContractError(contract.name, missing)
    return frame.loc[:, list(contract.columns)].astype(dict(contract.fields))


@dataclass(frozen=True, slots=True)
class OverviewReport:
    """Group seven plain result frames; fields are frozen, pandas frames mutable.

    Construction copies and normalizes inputs. Quality rows with a missing column
    label contain table duplicate metrics; other rows contain column metrics.
    Time min/max/span retain native scalar values, including ordinal numbers and
    datetime/timedelta values. Interval bounds use min/max and endpoint labels.
    """

    table_summary: pd.DataFrame = field(
        default_factory=partial(empty_result_frame, TABLE_SUMMARY_CONTRACT)
    )
    column_summary: pd.DataFrame = field(
        default_factory=partial(empty_result_frame, COLUMN_SUMMARY_CONTRACT)
    )
    identifier_summary: pd.DataFrame = field(
        default_factory=partial(empty_result_frame, IDENTIFIER_SUMMARY_CONTRACT)
    )
    grain_summary: pd.DataFrame = field(
        default_factory=partial(empty_result_frame, GRAIN_SUMMARY_CONTRACT)
    )
    time_summary: pd.DataFrame = field(
        default_factory=partial(empty_result_frame, TIME_SUMMARY_CONTRACT)
    )
    quality_summary: pd.DataFrame = field(
        default_factory=partial(empty_result_frame, QUALITY_SUMMARY_CONTRACT)
    )
    provenance: pd.DataFrame = field(
        default_factory=partial(empty_result_frame, PROVENANCE_CONTRACT)
    )

    def __post_init__(self) -> None:
        for contract in RESULT_CONTRACTS:
            frame = normalize_result_frame(getattr(self, contract.name), contract)
            object.__setattr__(self, contract.name, frame)

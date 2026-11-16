"""Overview package for profiling structure, grain, keys, time, and quality."""

from propensity.overview.dataset import profile_dataset
from propensity.overview.dunnhumby import build_dunnhumby_overview
from propensity.overview.keys import profile_candidate_grain, profile_identifiers
from propensity.overview.missing import blank_mask, missing_mask
from propensity.overview.models import (
    COLUMN_SUMMARY_COLUMNS,
    COLUMN_SUMMARY_CONTRACT,
    GRAIN_SUMMARY_COLUMNS,
    GRAIN_SUMMARY_CONTRACT,
    IDENTIFIER_SUMMARY_COLUMNS,
    IDENTIFIER_SUMMARY_CONTRACT,
    NUMERIC_SUMMARY_COLUMNS,
    NUMERIC_SUMMARY_CONTRACT,
    PROVENANCE_COLUMNS,
    PROVENANCE_CONTRACT,
    QUALITY_SUMMARY_COLUMNS,
    QUALITY_SUMMARY_CONTRACT,
    RESULT_CONTRACTS,
    TABLE_SUMMARY_COLUMNS,
    TABLE_SUMMARY_CONTRACT,
    TIME_SUMMARY_COLUMNS,
    TIME_SUMMARY_CONTRACT,
    OverviewContractError,
    OverviewReport,
    ResultContract,
    empty_result_frame,
    normalize_result_frame,
)
from propensity.overview.numeric import profile_numeric
from propensity.overview.preview import preview_dunnhumby_tables, preview_table
from propensity.overview.quality import profile_quality
from propensity.overview.structure import profile_columns, profile_table_structure
from propensity.overview.temporal import profile_time_coverage

__all__ = [
    "COLUMN_SUMMARY_COLUMNS",
    "COLUMN_SUMMARY_CONTRACT",
    "GRAIN_SUMMARY_COLUMNS",
    "GRAIN_SUMMARY_CONTRACT",
    "IDENTIFIER_SUMMARY_COLUMNS",
    "IDENTIFIER_SUMMARY_CONTRACT",
    "NUMERIC_SUMMARY_COLUMNS",
    "NUMERIC_SUMMARY_CONTRACT",
    "PROVENANCE_COLUMNS",
    "PROVENANCE_CONTRACT",
    "QUALITY_SUMMARY_COLUMNS",
    "QUALITY_SUMMARY_CONTRACT",
    "RESULT_CONTRACTS",
    "TABLE_SUMMARY_COLUMNS",
    "TABLE_SUMMARY_CONTRACT",
    "TIME_SUMMARY_COLUMNS",
    "TIME_SUMMARY_CONTRACT",
    "OverviewContractError",
    "OverviewReport",
    "ResultContract",
    "blank_mask",
    "build_dunnhumby_overview",
    "empty_result_frame",
    "missing_mask",
    "normalize_result_frame",
    "preview_dunnhumby_tables",
    "preview_table",
    "profile_candidate_grain",
    "profile_columns",
    "profile_dataset",
    "profile_identifiers",
    "profile_numeric",
    "profile_quality",
    "profile_table_structure",
    "profile_time_coverage",
]

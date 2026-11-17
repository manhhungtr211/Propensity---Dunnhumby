"""Compatibility imports for temporal analysis."""

from propensity.eda._expressions import date_expr as _date_expr

from propensity.eda import (
    AnalysisColumnError,
    TemporalReport,
    describe_temporal,
)

__all__ = [
    "AnalysisColumnError",
    "TemporalReport",
    "_date_expr",
    "describe_temporal",
]

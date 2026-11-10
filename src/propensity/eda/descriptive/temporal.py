"""Compatibility imports for temporal analysis."""

from propensity.eda import (
    AnalysisColumnError,
    TemporalReport,
    describe_temporal,
)
from propensity.eda._expressions import date_expr as _date_expr

__all__ = [
    "AnalysisColumnError",
    "TemporalReport",
    "_date_expr",
    "describe_temporal",
]

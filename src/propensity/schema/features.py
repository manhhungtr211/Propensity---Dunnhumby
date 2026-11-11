"""Profiling routes that cannot always be inferred from physical dtypes."""

from __future__ import annotations

from enum import StrEnum, unique


@unique
class AnalysisKind(StrEnum):
    """Computation family appropriate for a feature."""

    IDENTIFIER = "identifier"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    ORDINAL_TIME = "ordinal_time"
    CLOCK_TIME = "clock_time"

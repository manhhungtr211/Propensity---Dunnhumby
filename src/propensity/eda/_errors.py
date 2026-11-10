"""Internal error definitions shared by EDA modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AnalysisColumnError(Exception):
    """Raised when a required analysis column is absent or unsupported."""

    column: str
    reason: str

    def __str__(self) -> str:
        return f"Cột phân tích '{self.column}' {self.reason}"


@dataclass(frozen=True, slots=True)
class AnalysisConfigurationError(ValueError):
    """Raised when an analysis call lacks required configuration."""

    reason: str

    def __str__(self) -> str:
        return self.reason

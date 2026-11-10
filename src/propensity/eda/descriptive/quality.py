"""Compatibility imports for data-quality analysis."""

from propensity.eda import QualityReport, describe_quality
from propensity.eda._expressions import missing_condition as _missing_condition

__all__ = ["QualityReport", "_missing_condition", "describe_quality"]

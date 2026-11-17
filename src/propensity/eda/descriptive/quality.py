"""Compatibility imports for data-quality analysis."""

from propensity.eda._expressions import missing_condition as _missing_condition

from propensity.eda import QualityReport, describe_quality

__all__ = ["QualityReport", "_missing_condition", "describe_quality"]

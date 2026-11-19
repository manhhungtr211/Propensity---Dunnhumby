"""Reusable data-understanding tools for propensity projects."""

from propensity.labels import (
    DEFAULT_PURCHASE_LABEL_SPEC,
    PurchaseLabelError,
    PurchaseLabelSpec,
    build_purchase_labels,
)
from propensity.overview import (
    OverviewReport,
    build_dunnhumby_overview,
    profile_dataset,
)
from propensity.purchases import PurchaseValidationError, valid_purchase_lines
from propensity.schema import ColumnRole, DatasetSpec, DatasetSpecError

__all__ = [
    "DEFAULT_PURCHASE_LABEL_SPEC",
    "ColumnRole",
    "DatasetSpec",
    "DatasetSpecError",
    "OverviewReport",
    "PurchaseLabelError",
    "PurchaseLabelSpec",
    "PurchaseValidationError",
    "build_dunnhumby_overview",
    "build_purchase_labels",
    "profile_dataset",
    "valid_purchase_lines",
]

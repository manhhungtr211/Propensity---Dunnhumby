"""Compatibility imports for dataset inventory helpers."""

from src.eda.inventory import (
    DatasetInventory,
    column_overview,
    describe_dataset,
    dtype_groups,
    preview,
    shape,
)

__all__ = [
    "DatasetInventory",
    "column_overview",
    "describe_dataset",
    "dtype_groups",
    "preview",
    "shape",
]

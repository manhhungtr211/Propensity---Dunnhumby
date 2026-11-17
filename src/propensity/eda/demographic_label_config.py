"""Configuration for demographic-label analysis."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

FEATURES: Final[tuple[str, ...]] = (
    "AGE_DESC",
    "MARITAL_STATUS_CODE",
    "INCOME_DESC",
    "HOMEOWNER_DESC",
    "HH_COMP_DESC",
    "HOUSEHOLD_SIZE_DESC",
    "KID_CATEGORY_DESC",
)
MISSING_GROUP: Final = "NO_DEMOGRAPHIC_RECORD"
LABEL_NAMES: Final[dict[int, str]] = {0: "Negative", 1: "Positive"}
CATEGORY_ORDERS: Final[Mapping[str, tuple[str, ...]]] = {
    "AGE_DESC": ("19-24", "25-34", "35-44", "45-54", "55-64", "65+"),
    "MARITAL_STATUS_CODE": ("A", "B", "U"),
    "INCOME_DESC": (
        "Under 15K",
        "15-24K",
        "25-34K",
        "35-49K",
        "50-74K",
        "75-99K",
        "100-124K",
        "125-149K",
        "150-174K",
        "175-199K",
        "200-249K",
        "250K+",
    ),
    "HOMEOWNER_DESC": (
        "Homeowner",
        "Probable Owner",
        "Renter",
        "Probable Renter",
        "Unknown",
    ),
    "HH_COMP_DESC": (
        "Single Female",
        "Single Male",
        "1 Adult Kids",
        "2 Adults No Kids",
        "2 Adults Kids",
        "Unknown",
    ),
    "HOUSEHOLD_SIZE_DESC": ("1", "2", "3", "4", "5+"),
    "KID_CATEGORY_DESC": ("None/Unknown", "1", "2", "3+"),
}

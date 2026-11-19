"""Shared purchase validation and filtering rules."""

from __future__ import annotations

from typing import Final
import pandas as pd

_REQUIRED_COLUMNS: Final = ("SALES_VALUE",)


class PurchaseValidationError(ValueError):
    """Raised when transaction data violates purchase validation requirements."""


def valid_purchase_lines(transactions: pd.DataFrame) -> pd.DataFrame:
    """Filter transactions to valid purchase lines (SALES_VALUE > 0).

    A valid purchase line represents a real commercial exchange where
    positive revenue was realized. Zero-price lines (e.g. free promotional gifts)
    and negative lines (e.g. returns/refunds) are excluded.
    """
    if "SALES_VALUE" not in transactions.columns:
        raise PurchaseValidationError(
            f"Thiếu cột bắt buộc để xác thực purchase: 'SALES_VALUE'. "
            f"Các cột hiện có: {list(transactions.columns)}"
        )
    return transactions.loc[transactions["SALES_VALUE"] > 0].copy()

"""Expressions shared by encoded clock-time profiling."""

from __future__ import annotations

import polars as pl


def valid_clock_expr(column: str) -> pl.Expr:
    """Return whether an integer-encoded HHMM value is possible."""
    value = pl.col(column)
    return (value >= 0) & ((value // 100) <= 23) & ((value % 100) <= 59)

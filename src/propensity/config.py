"""Defaults shared by reusable analysis modules."""

from __future__ import annotations

from typing import Final

DEFAULT_TOP_N: Final = 20
DEFAULT_TIME_FREQ: Final = "1mo"
DEFAULT_NA_TOKENS: Final = (
    "",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "?",
)

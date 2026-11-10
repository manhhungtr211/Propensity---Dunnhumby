"""Shared runtime paths and defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
RAW_DATA_DIR: Final = PROJECT_ROOT / "data" / "raw"

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

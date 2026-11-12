"""Shared drawing helpers for transaction-behavior charts."""

from __future__ import annotations

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

from propensity.eda.visualization.plots import PlotInputError

BASE = "#9CB8C8"
BASE_DARK = "#3F657A"
BASE_LIGHT = "#DCE8EE"
HIGHLIGHT = "#D97706"
HIGHLIGHT_DARK = "#9A4B0A"
REFERENCE = "#7B8794"
INK = "#1F2937"
DENSITY_CMAP = "Blues"
CORRELATION_CMAP = LinearSegmentedColormap.from_list(
    "transaction_behavior_diverging", (BASE_DARK, "#F7F7F5", HIGHLIGHT_DARK)
)
PATTERN_COLORS = (REFERENCE, "#E8B786", BASE_DARK, HIGHLIGHT)


def require_columns(frame: pl.DataFrame, required: tuple[str, ...]) -> None:
    """Raise a chart-specific error when required analytical columns are absent."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise PlotInputError(f"Thiếu cột phục vụ biểu đồ: {missing}")


def numeric_values(frame: pl.DataFrame, column: str) -> np.ndarray:
    """Return finite, non-null values for one plotting column."""
    values = frame[column].drop_nulls().cast(pl.Float64).to_numpy()
    return values[np.isfinite(values)]


def finish_axis(axis: Axes) -> None:
    """Apply the shared minimal visual style."""
    axis.grid(axis="y", color=BASE_LIGHT, linewidth=0.8, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)

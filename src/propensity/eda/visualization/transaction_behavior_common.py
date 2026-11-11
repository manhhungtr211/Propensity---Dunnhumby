"""Shared drawing helpers for transaction-behavior charts."""

from __future__ import annotations

import numpy as np
import polars as pl
from matplotlib.axes import Axes

from propensity.eda.visualization.plots import PlotInputError

GREEN = "#2F6B4F"
ORANGE = "#D97706"
BLUE = "#2563EB"
PURPLE = "#7C3AED"
GRAY = "#6B7280"


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
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.6)
    axis.spines[["top", "right"]].set_visible(False)

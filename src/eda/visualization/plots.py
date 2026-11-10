"""Reusable charts for descriptive propensity-data analysis."""

from __future__ import annotations

from math import ceil, sqrt
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter, StrMethodFormatter

from src.config import DEFAULT_TOP_N
from src.eda.descriptive.quality import describe_quality

Frame = pl.DataFrame | pl.LazyFrame


class PlotInputError(Exception):
    """Raised when chart input does not satisfy the plotting contract."""


def _require_columns(frame: pl.DataFrame, required: tuple[str, ...]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise PlotInputError(f"Thiếu cột phục vụ biểu đồ: {missing}")


def plot_records_over_time(monthly_records: pl.DataFrame) -> Figure:
    """Plot monthly record counts from ``TemporalReport['monthly_records']``."""
    _require_columns(monthly_records, ("month", "n_records"))
    data = monthly_records.sort("month")
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    axis.plot(
        data["month"].to_list(), data["n_records"].to_list(), marker="o", linewidth=2
    )
    axis.set(
        title="Số lượng bản ghi theo thời gian", xlabel="Tháng", ylabel="Số bản ghi"
    )
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.grid(axis="y", alpha=0.25)
    figure.autofmt_xdate()
    return figure


def plot_records_per_entity(
    entity_history: pl.DataFrame,
    *,
    binning: Literal["auto", "linear", "log"] = "auto",
    bins: int | None = None,
    log_y: bool = False,
) -> Figure:
    """Plot records per entity, grouping sparse long tails into wider bins.

    ``binning="auto"`` uses power-of-two intervals for a long-tailed
    distribution and a regular histogram otherwise. Use ``binning="log"``
    to force intervals such as 1, 2--3, 4--7, and 8--15.
    """
    _require_columns(entity_history, ("n_records",))
    values = entity_history["n_records"].drop_nulls().to_numpy()
    if values.size == 0:
        raise PlotInputError("Không có giá trị n_records hợp lệ để vẽ")
    if np.any(values < 1):
        raise PlotInputError("n_records phải lớn hơn hoặc bằng 1")
    if binning not in {"auto", "linear", "log"}:
        raise PlotInputError("binning phải là 'auto', 'linear' hoặc 'log'")
    if bins is not None and bins <= 0:
        raise PlotInputError("bins phải lớn hơn 0")

    maximum = int(values.max())
    median = float(np.median(values))
    long_tail = maximum >= max(16, 8 * median)
    use_log_bins = binning == "log" or (binning == "auto" and long_tail)

    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    if use_log_bins:
        upper_limits: list[int] = []
        upper = 1
        while upper < maximum:
            upper_limits.append(upper)
            upper = 2 * upper + 1
        upper_limits.append(upper)
        edges = np.array([0.5, *(limit + 0.5 for limit in upper_limits)])
        counts, _ = np.histogram(values, bins=edges)
        lower_limits = [1, *(limit + 1 for limit in upper_limits[:-1])]
        labels = [
            str(lower) if lower == upper else f"{lower}–{upper}"
            for lower, upper in zip(lower_limits, upper_limits, strict=True)
        ]
        positions = np.arange(len(labels))
        axis.bar(positions, counts, color="#2563EB", edgecolor="white")
        axis.set_xticks(positions, labels, rotation=30, ha="right")
        median_bin = int(np.searchsorted(upper_limits, median, side="left"))
        axis.axvline(
            median_bin,
            color="#DC2626",
            linestyle="--",
            label=f"Median = {median:,.1f}",
        )
        x_label = "Số bản ghi/customer"
    else:
        linear_bins = bins or min(50, max(1, ceil(sqrt(values.size))))
        axis.hist(values, bins=linear_bins, color="#2563EB", edgecolor="white")
        axis.axvline(
            median,
            color="#DC2626",
            linestyle="--",
            label=f"Median = {median:,.1f}",
        )
        x_label = "Số bản ghi/customer"
    if log_y:
        axis.set_yscale("log")
    axis.set(
        title="Phân phối số bản ghi trên mỗi khách hàng",
        xlabel=x_label,
        ylabel="Số khách hàng" + (" (thang log)" if log_y else ""),
    )
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    return figure


def plot_active_periods_per_entity(entity_history: pl.DataFrame) -> Figure:
    """Plot entities by number of distinct active months, not consecutive months."""
    _require_columns(entity_history, ("n_active_months",))
    values = entity_history["n_active_months"].drop_nulls().to_numpy()
    if values.size == 0:
        raise PlotInputError("Không có giá trị n_active_months hợp lệ để vẽ")

    labels = ["0", "1", "2–3", "4–6", "7–12", "13–18", "19–24", "25+"]
    bins = np.array([-0.5, 0.5, 1.5, 3.5, 6.5, 12.5, 18.5, 24.5, np.inf])
    counts, _ = np.histogram(values, bins=bins)
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    axis.bar(labels, counts, color="#0F766E")
    axis.set(
        title="Số kỳ hoạt động của mỗi khách hàng",
        xlabel="Số tháng riêng biệt có hoạt động",
        ylabel="Số khách hàng",
    )
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.grid(axis="y", alpha=0.25)
    return figure


def plot_missing_values(df: Frame, top_n: int = DEFAULT_TOP_N) -> Figure:
    """Plot the features with highest missing percentages."""
    if top_n <= 0:
        raise PlotInputError("top_n phải lớn hơn 0")
    eager = df.collect() if isinstance(df, pl.LazyFrame) else df
    if {"column", "missing_pct"}.issubset(eager.columns):
        quality = eager.select("column", "missing_pct")
    else:
        quality = describe_quality(eager.lazy())["column_quality"].select(
            "column",
            "missing_pct",
        )
    data = quality.sort("missing_pct").tail(top_n)
    figure, axis = plt.subplots(
        figsize=(10, max(4.5, 0.35 * data.height + 1.5)),
        constrained_layout=True,
    )
    axis.barh(data["column"].to_list(), data["missing_pct"].to_list(), color="#D97706")
    axis.set(
        title="Các feature có tỷ lệ missing cao nhất",
        xlabel="Tỷ lệ missing",
        ylabel="Feature",
    )
    axis.set_xlim(0, 100)
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    axis.grid(axis="x", alpha=0.25)
    return figure

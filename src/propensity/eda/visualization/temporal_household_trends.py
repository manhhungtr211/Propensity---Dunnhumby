"""Rolling-window views of household behavior change."""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from propensity.eda.visualization.transaction_behavior_common import (
    BASE,
    BASE_DARK,
    DENSITY_CMAP,
    HIGHLIGHT,
    INK,
    REFERENCE,
    finish_axis,
    require_columns,
)

CHANGE_BANDS: Final = (
    ("Nhanh hơn\n(< 0,8×)", BASE_DARK),
    ("Ổn định\n(0,8–1,25×)", BASE),
    ("Chậm lại\n(1,25–2×)", "#E9A85B"),
    ("Chậm mạnh\n(≥ 2×)", HIGHLIGHT),
)


def _population_trend(windows: pl.DataFrame) -> pl.DataFrame:
    reliable = pl.col("n_intervals").fill_null(0) >= 3
    return (
        windows.group_by("window_start_week", "window_end_week")
        .agg(
            pl.len().alias("n_active_households"),
            pl.col("baskets_per_week").median().alias("median_baskets_per_week"),
            pl.col("median_interval_days")
            .filter(reliable)
            .median()
            .alias("median_interval_days"),
            pl.col("interval_cv")
            .filter(reliable)
            .median()
            .alias("median_interval_cv"),
        )
        .sort("window_end_week")
    )


def plot_household_behavior_change(windows: pl.DataFrame) -> Figure:
    """Show rolling trends and the latest non-overlapping cadence change."""
    require_columns(
        windows,
        (
            "window_start_week",
            "window_end_week",
            "window_weeks",
            "n_baskets",
            "baskets_per_week",
            "n_intervals",
            "median_interval_days",
            "interval_cv",
        ),
    )
    trend = _population_trend(windows)
    max_active = int(trend["n_active_households"].max())
    mature_start = trend.height // 2
    mature_basket_rate = float(
        trend.slice(mature_start)["median_baskets_per_week"].median()
    )
    baseline = trend.filter(
        (pl.col("n_active_households") >= max_active * 0.9)
        & (
            pl.col("median_baskets_per_week")
            >= mature_basket_rate * 0.9
        )
    ).row(0, named=True)
    baseline_week = int(baseline["window_end_week"])
    indexed = trend.with_columns(
        (
            pl.col("median_baskets_per_week")
            / float(baseline["median_baskets_per_week"])
            * 100
        ).alias("basket_rate_index"),
        (
            pl.col("median_interval_days")
            / float(baseline["median_interval_days"])
            * 100
        ).alias("interval_index"),
        (
            pl.col("median_interval_cv")
            / float(baseline["median_interval_cv"])
            * 100
        ).alias("cv_index"),
    )

    window_weeks = int(windows["window_weeks"][0])
    recent_end = int(windows["window_end_week"].max())
    recent_start = recent_end - window_weeks + 1
    prior_end = recent_start - 1
    prior_start = prior_end - window_weeks + 1
    entity_col = windows.columns[0]
    prior = windows.filter(
        (pl.col("window_start_week") == prior_start)
        & (pl.col("n_intervals") >= 3)
    ).select(
        pl.exclude(
            entity_col,
            "window_start_week",
            "window_end_week",
            "window_weeks",
        ).name.suffix("_prior"),
        pl.col(entity_col),
    )
    recent = windows.filter(
        (pl.col("window_start_week") == recent_start)
        & (pl.col("n_intervals") >= 3)
    ).select(
        pl.exclude(
            entity_col,
            "window_start_week",
            "window_end_week",
            "window_weeks",
        ).name.suffix("_recent"),
        pl.col(entity_col),
    )
    comparison = prior.join(recent, on=entity_col, how="inner").with_columns(
        (
            pl.col("median_interval_days_recent")
            / pl.col("median_interval_days_prior")
        ).alias("interval_change_ratio")
    )

    figure = plt.figure(figsize=(15, 10.2), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(1.05, 1),
        hspace=0.42,
        wspace=0.20,
    )
    trend_axis = figure.add_subplot(grid[0, :])
    comparison_axis = figure.add_subplot(grid[1, 0])
    state_axis = figure.add_subplot(grid[1, 1])

    trend_axis.axvspan(
        float(indexed["window_end_week"].min()),
        baseline_week,
        color="#F3F4F6",
        alpha=0.8,
        label="Onboarding / coverage build-up",
    )
    trend_axis.plot(
        indexed["window_end_week"],
        indexed["basket_rate_index"],
        color=BASE_DARK,
        linewidth=2.5,
        marker="o",
        label="Basket rate (↑ = mua nhiều hơn)",
    )
    trend_axis.plot(
        indexed["window_end_week"],
        indexed["interval_index"],
        color=HIGHLIGHT,
        linewidth=2.5,
        marker="o",
        label="Median interval (↑ = chậm hơn)",
    )
    trend_axis.plot(
        indexed["window_end_week"],
        indexed["cv_index"],
        color=REFERENCE,
        linewidth=2.2,
        marker="o",
        label="Median CV (↑ = thất thường hơn)",
    )
    trend_axis.axhline(100, color=INK, linewidth=1, linestyle="--", alpha=0.6)
    trend_axis.axvline(
        baseline_week,
        color=INK,
        linewidth=1,
        linestyle=":",
    )
    trend_axis.set(
        title="Rolling 13 tuần, cập nhật mỗi 4 tuần",
        xlabel="Tuần kết thúc window",
        ylabel=f"Chỉ số tương đối (tuần {baseline_week} = 100)",
    )
    trend_axis.legend(frameon=False, ncols=2, loc="lower right")

    prior_values = comparison["median_interval_days_prior"].to_numpy()
    recent_values = comparison["median_interval_days_recent"].to_numpy()
    comparison_axis.hexbin(
        prior_values,
        recent_values,
        gridsize=28,
        mincnt=1,
        bins="log",
        xscale="log",
        yscale="log",
        cmap=DENSITY_CMAP,
    )
    diagonal_min = float(min(prior_values.min(), recent_values.min()))
    diagonal_max = float(max(prior_values.max(), recent_values.max()))
    comparison_axis.plot(
        [diagonal_min, diagonal_max],
        [diagonal_min, diagonal_max],
        color=HIGHLIGHT,
        linewidth=1.8,
        linestyle="--",
        label="Không đổi (Y = X)",
    )
    comparison_axis.set(
        title=f"Cadence tuần {prior_start}–{prior_end} vs {recent_start}–{recent_end}",
        xlabel="Median interval — prior (ngày, log)",
        ylabel="Median interval — recent (ngày, log)",
    )
    comparison_axis.legend(frameon=False, loc="upper left")

    ratios = comparison["interval_change_ratio"].to_numpy()
    state_counts = np.asarray(
        [
            int((ratios < 0.8).sum()),
            int(((ratios >= 0.8) & (ratios < 1.25)).sum()),
            int(((ratios >= 1.25) & (ratios < 2)).sum()),
            int((ratios >= 2).sum()),
        ]
    )
    state_shares = state_counts / state_counts.sum()
    state_axis.bar(
        [label for label, _color in CHANGE_BANDS],
        state_shares,
        color=[color for _label, color in CHANGE_BANDS],
        edgecolor="white",
        width=0.74,
    )
    for index, share in enumerate(state_shares):
        state_axis.text(
            index,
            share + 0.012,
            f"{share:.1%}\n({state_counts[index]:,})",
            ha="center",
            va="bottom",
            color=INK,
        )
    state_axis.yaxis.set_major_formatter(PercentFormatter(1))
    state_axis.set(
        title="Household thay đổi cadence gần đây như thế nào?",
        xlabel="Recent median / prior median",
        ylabel="Tỷ lệ household đủ evidence",
        ylim=(0, max(0.55, float(state_shares.max()) + 0.10)),
    )

    for axis in (trend_axis, comparison_axis, state_axis):
        finish_axis(axis)
    figure.subplots_adjust(
        top=0.84,
        bottom=0.09,
        left=0.07,
        right=0.98,
    )
    figure.suptitle(
        "Household behavior thay đổi như thế nào theo thời gian?",
        fontsize=18,
        fontweight="bold",
        color=INK,
        y=0.97,
    )
    figure.text(
        0.5,
        0.925,
        (
            "Rolling windows dùng để nhìn trend; hai window 13 tuần không "
            "chồng lấn dùng để phân loại thay đổi gần nhất"
        ),
        ha="center",
        color="#4B5563",
        fontsize=10.5,
    )
    return figure

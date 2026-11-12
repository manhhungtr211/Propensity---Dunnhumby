"""Historical-coverage evidence for choosing a behavioral lookback."""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from propensity.eda.visualization.transaction_behavior_common import (
    BASE_DARK,
    DENSITY_CMAP,
    INK,
    finish_axis,
    require_columns,
)

COVERAGE_METRICS: Final = (
    ("share_ge_1_basket", "Có ≥1 basket"),
    ("share_ge_2_baskets", "Có ≥2 baskets"),
    ("share_ge_5_baskets", "Có ≥5 baskets"),
    ("share_gap_ready", "Đủ tính purchase gap\n(≥2 active days)"),
    ("share_ge_2_active_weeks", "Hoạt động ≥2 tuần"),
)


def plot_historical_activity_coverage(coverage: pl.DataFrame) -> Figure:
    """Draw coverage percentages and median basket evidence by lookback."""
    require_columns(
        coverage,
        (
            "cutoff_day",
            "window_days",
            "n_eligible_households",
            "median_basket_count",
            *(column for column, _label in COVERAGE_METRICS),
        ),
    )
    ordered = coverage.sort("window_days")
    windows = ordered["window_days"].to_list()
    matrix = np.asarray(
        [ordered[column].to_numpy() for column, _label in COVERAGE_METRICS]
    )

    figure, (heatmap_axis, median_axis) = plt.subplots(
        1,
        2,
        figsize=(15, 6.8),
        constrained_layout=False,
        gridspec_kw={"width_ratios": (1.65, 1)},
    )
    image = heatmap_axis.imshow(
        matrix,
        cmap=DENSITY_CMAP,
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    heatmap_axis.set_xticks(
        np.arange(len(windows)),
        [f"{window} ngày" for window in windows],
    )
    heatmap_axis.set_yticks(
        np.arange(len(COVERAGE_METRICS)),
        [label for _column, label in COVERAGE_METRICS],
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            heatmap_axis.text(
                column_index,
                row_index,
                f"{value:.1%}",
                ha="center",
                va="center",
                color="white" if value >= 0.58 else INK,
                fontweight="bold",
            )
    heatmap_axis.set(
        title="Household coverage theo độ dài lịch sử",
        xlabel="Trailing window tại cùng cutoff",
    )
    heatmap_axis.tick_params(length=0)
    for spine in heatmap_axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=heatmap_axis, fraction=0.035, pad=0.03)
    colorbar.set_label("Tỷ lệ household")

    medians = ordered["median_basket_count"].to_numpy()
    bars = median_axis.bar(
        [str(window) for window in windows],
        medians,
        color=BASE_DARK,
        edgecolor="white",
        width=0.72,
    )
    for bar, value in zip(bars, medians, strict=True):
        median_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(medians) * 0.025,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            color=INK,
            fontweight="bold",
        )
    median_axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    median_axis.set(
        title="Median basket count (bao gồm zero)",
        xlabel="Lookback T (ngày)",
        ylabel="Median baskets / household",
        ylim=(0, max(medians) * 1.16),
    )
    finish_axis(median_axis)

    cutoff = int(ordered["cutoff_day"][0])
    n_households = int(ordered["n_eligible_households"][0])
    figure.subplots_adjust(
        top=0.79,
        bottom=0.13,
        left=0.15,
        right=0.98,
        wspace=0.25,
    )
    figure.suptitle(
        "Historical activity coverage: bao nhiêu lịch sử là đủ?",
        fontsize=18,
        fontweight="bold",
        color=INK,
        y=0.97,
    )
    figure.text(
        0.5,
        0.91,
        (
            f"Cùng {n_households:,} household tại cutoff DAY {cutoff}; "
            "household không mua trong window vẫn được giữ với basket count = 0"
        ),
        ha="center",
        color="#4B5563",
        fontsize=10.5,
    )
    return figure

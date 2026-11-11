"""Visual evidence for technical transaction-table EDA."""

from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.figure import Figure

from propensity.eda.visualization.plots import PlotInputError


def plot_return_interval_distribution(distribution: pl.DataFrame) -> Figure:
    """Plot short-term frequency and the full cumulative return-interval tail."""
    required = {
        "return_interval_days",
        "n_returns",
        "share",
        "cumulative_share",
    }
    missing = required.difference(distribution.columns)
    if missing:
        raise PlotInputError(f"Thiếu cột phục vụ biểu đồ: {sorted(missing)}")

    short_tail = distribution.filter(pl.col("return_interval_days") <= 60)
    median_day = distribution.filter(pl.col("cumulative_share") >= 0.5)[
        "return_interval_days"
    ].item(0)
    share_within_3 = distribution.filter(pl.col("return_interval_days") <= 3)[
        "share"
    ].sum()
    share_within_7 = distribution.filter(pl.col("return_interval_days") <= 7)[
        "share"
    ].sum()
    share_within_14 = distribution.filter(pl.col("return_interval_days") <= 14)[
        "share"
    ].sum()

    figure, (histogram_axis, cumulative_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 6.75),
        gridspec_kw={"width_ratios": (1.45, 1)},
    )
    histogram_axis.bar(
        short_tail["return_interval_days"].to_list(),
        (short_tail["share"] * 100).to_list(),
        width=0.85,
        color="#2F6B4F",
        edgecolor="none",
    )
    histogram_axis.axvline(
        median_day,
        color="#D97706",
        linewidth=2,
        linestyle="--",
        label=f"Median = {median_day} ngày",
    )
    histogram_axis.set(
        title="Phân bố trong 60 ngày đầu",
        xlabel="Số ngày từ lần mua trước",
        ylabel="Tỷ trọng lượt quay lại (%)",
        xlim=(0, 61),
    )
    histogram_axis.set_xticks((1, 7, 14, 21, 30, 45, 60))
    histogram_axis.legend(frameon=False, loc="upper right")

    cumulative_axis.plot(
        distribution["return_interval_days"].to_list(),
        (distribution["cumulative_share"] * 100).to_list(),
        color="#2F6B4F",
        linewidth=2.5,
    )
    cumulative_axis.set_xscale("log")
    cumulative_axis.set(
        title="Tích lũy trên toàn bộ đuôi phân phối",
        xlabel="Số ngày từ lần mua trước (log scale)",
        ylabel="Tỷ lệ tích lũy (%)",
        ylim=(0, 102),
    )
    cumulative_axis.set_yticks((0, 25, 50, 75, 90, 100))
    cumulative_axis.text(
        0.04,
        0.43,
        (
            f"≤ 3 ngày: {share_within_3:.1%}\n"
            f"≤ 7 ngày: {share_within_7:.1%}\n"
            f"≤ 14 ngày: {share_within_14:.1%}"
        ),
        transform=cumulative_axis.transAxes,
        fontsize=11,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": "#D1D5DB",
        },
    )
    for axis in (histogram_axis, cumulative_axis):
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "Một nửa số lượt quay lại diễn ra trong 3 ngày",
        fontsize=18,
        fontweight="bold",
        color="#173D2A",
        y=0.97,
    )
    figure.text(
        0.5,
        0.91,
        (
            "Khoảng quay lại = chênh lệch DAY giữa hai ngày mua liên tiếp của "
            "cùng household; nhiều basket cùng ngày chỉ tính một lần"
        ),
        ha="center",
        fontsize=10.5,
        color="#4B5563",
    )
    figure.subplots_adjust(
        top=0.82,
        bottom=0.14,
        left=0.08,
        right=0.97,
        wspace=0.28,
    )
    return figure

"""Lifecycle and exposure views for household temporal behavior."""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter

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

ENTRY_WINDOWS: Final = (
    ("Trước 90 ngày cuối", "#D7DCE0"),
    ("61–90 ngày cuối", "#F3D6A7"),
    ("31–60 ngày cuối", "#E9A85B"),
    ("0–30 ngày cuối", HIGHLIGHT),
)
RECENCY_WINDOWS: Final = (
    ("Trên 90 ngày", "#D7DCE0"),
    ("61–90 ngày", "#BFD4DF"),
    ("31–60 ngày", "#7FA5B9"),
    ("0–30 ngày", BASE_DARK),
)
LATENESS_BANDS: Final = (
    ("Chưa đến nhịp riêng\n(< 1×)", "#D7DCE0"),
    ("Hơi trễ\n(1–2×)", "#7FA5B9"),
    ("Trễ rõ\n(2–4×)", "#E9A85B"),
    ("Rất trễ\n(> 4×)", HIGHLIGHT),
)


def _binned_medians(
    exposure: np.ndarray,
    baskets: np.ndarray,
    *,
    n_bins: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(float(exposure.min()), float(exposure.max()), n_bins + 1)
    bin_ids = np.clip(np.digitize(exposure, edges[1:-1]), 0, n_bins - 1)
    centers: list[float] = []
    medians: list[float] = []
    for bin_id in range(n_bins):
        selected = bin_ids == bin_id
        if int(selected.sum()) < 5:
            continue
        centers.append(float(np.median(exposure[selected])))
        medians.append(float(np.median(baskets[selected])))
    return np.asarray(centers), np.asarray(medians)


def plot_lifecycle_coverage_exposure(households: pl.DataFrame) -> Figure:
    """Show entry cohorts beside the exposure-to-basket relationship."""
    require_columns(
        households,
        (
            "first_purchase_week",
            "days_since_first_purchase",
            "exposure_days",
            "basket_count",
        ),
    )
    days_since_entry = pl.col("days_since_first_purchase")
    labelled = households.with_columns(
        pl.when(days_since_entry <= 30)
        .then(pl.lit("0–30 ngày cuối"))
        .when(days_since_entry <= 60)
        .then(pl.lit("31–60 ngày cuối"))
        .when(days_since_entry <= 90)
        .then(pl.lit("61–90 ngày cuối"))
        .otherwise(pl.lit("Trước 90 ngày cuối"))
        .alias("entry_window")
    )
    cohort_counts = (
        labelled.group_by("first_purchase_week", "entry_window")
        .agg(pl.len().alias("n_households"))
        .sort("first_purchase_week")
    )
    weeks = np.sort(labelled["first_purchase_week"].unique().to_numpy())

    figure, (cohort_axis, exposure_axis) = plt.subplots(
        1,
        2,
        figsize=(15, 6.8),
        constrained_layout=False,
        gridspec_kw={"width_ratios": (1.08, 1)},
    )
    bottom = np.zeros(weeks.size)
    for label, color in ENTRY_WINDOWS:
        counts_by_week = dict(
            cohort_counts.filter(pl.col("entry_window") == label)
            .select("first_purchase_week", "n_households")
            .iter_rows()
        )
        counts = np.asarray([counts_by_week.get(int(week), 0) for week in weeks])
        cohort_axis.bar(
            weeks,
            counts,
            bottom=bottom,
            width=0.86,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            label=label,
        )
        bottom += counts
    cohort_axis.set(
        title="Household gia nhập theo cohort tuần",
        xlabel="Tuần mua đầu tiên",
        ylabel="Số household mới",
    )
    cohort_axis.legend(frameon=False, fontsize=9, loc="upper right")
    entry_ages = labelled["days_since_first_purchase"].to_numpy()
    recent_counts = tuple(int((entry_ages <= days).sum()) for days in (30, 60, 90))
    recent_shares = tuple(count / entry_ages.size for count in recent_counts)
    cohort_axis.text(
        0.97,
        0.70,
        (
            f"≤ 30 ngày: {recent_counts[0]:,} ({recent_shares[0]:.2%})\n"
            f"≤ 60 ngày: {recent_counts[1]:,} ({recent_shares[1]:.2%})\n"
            f"≤ 90 ngày: {recent_counts[2]:,} ({recent_shares[2]:.2%})"
        ),
        transform=cohort_axis.transAxes,
        ha="right",
        va="top",
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#D7DCE0",
        },
    )
    peak_index = int(np.argmax(bottom))
    cohort_axis.annotate(
        f"Peak: tuần {int(weeks[peak_index])}\n{int(bottom[peak_index])} household",
        xy=(weeks[peak_index], bottom[peak_index]),
        xytext=(14, -18),
        textcoords="offset points",
        color=INK,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": REFERENCE},
    )

    exposure = labelled["exposure_days"].cast(pl.Float64).to_numpy()
    baskets = labelled["basket_count"].cast(pl.Float64).to_numpy()
    recent = labelled["days_since_first_purchase"].to_numpy() <= 30
    exposure_axis.scatter(
        exposure[~recent],
        baskets[~recent],
        s=18,
        color=REFERENCE,
        alpha=0.28,
        linewidths=0,
        label="Gia nhập trước 30 ngày cuối",
    )
    exposure_axis.scatter(
        exposure[recent],
        baskets[recent],
        s=28,
        color=HIGHLIGHT,
        alpha=0.82,
        linewidths=0,
        label="Gia nhập trong 30 ngày cuối",
    )
    median_x, median_y = _binned_medians(exposure, baskets)
    exposure_axis.plot(
        median_x,
        median_y,
        color=BASE_DARK,
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Median theo exposure bin",
    )
    spearman = float(
        labelled.select(
            pl.corr("exposure_days", "basket_count", method="spearman")
        ).item()
    )
    exposure_axis.text(
        0.04,
        0.95,
        f"Spearman ρ = {spearman:.2f}",
        transform=exposure_axis.transAxes,
        va="top",
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#D7DCE0",
        },
    )
    exposure_axis.set_yscale("log")
    exposure_axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{value:,.0f}")
    )
    exposure_axis.set(
        title="Exposure ảnh hưởng đến basket volume",
        xlabel="Số ngày được quan sát từ lần mua đầu",
        ylabel="Tổng số basket / household (log scale)",
    )
    exposure_axis.legend(frameon=False, fontsize=9, loc="lower right")

    for axis in (cohort_axis, exposure_axis):
        finish_axis(axis)
    figure.subplots_adjust(
        top=0.79,
        bottom=0.13,
        left=0.07,
        right=0.98,
        wspace=0.14,
    )
    figure.suptitle(
        "Kiểm tra exposure bias trong raw basket count",
        fontsize=17,
        fontweight="bold",
        color=INK,
        y=0.97,
    )
    figure.text(
        0.5,
        0.91,
        (
            "Cohort được xác định bằng lần mua đầu tiên; exposure tính đến "
            "snapshot và basket count là số basket distinct"
        ),
        ha="center",
        color="#4B5563",
        fontsize=10.5,
    )
    return figure


def plot_last_purchase_recency(households: pl.DataFrame) -> Figure:
    """Show last-purchase cohorts and the full household recency distribution."""
    require_columns(
        households,
        ("last_purchase_week", "recency_days"),
    )
    recency = pl.col("recency_days")
    labelled = households.with_columns(
        pl.when(recency <= 30)
        .then(pl.lit("0–30 ngày"))
        .when(recency <= 60)
        .then(pl.lit("31–60 ngày"))
        .when(recency <= 90)
        .then(pl.lit("61–90 ngày"))
        .otherwise(pl.lit("Trên 90 ngày"))
        .alias("recency_window")
    )
    cohort_counts = (
        labelled.group_by("last_purchase_week", "recency_window")
        .agg(pl.len().alias("n_households"))
        .sort("last_purchase_week")
    )
    weeks = np.sort(labelled["last_purchase_week"].unique().to_numpy())

    figure, (last_axis, recency_axis) = plt.subplots(
        1,
        2,
        figsize=(15, 6.8),
        constrained_layout=False,
        gridspec_kw={"width_ratios": (1.08, 1)},
    )
    bottom = np.zeros(weeks.size)
    for label, color in RECENCY_WINDOWS:
        counts_by_week = dict(
            cohort_counts.filter(pl.col("recency_window") == label)
            .select("last_purchase_week", "n_households")
            .iter_rows()
        )
        counts = np.asarray([counts_by_week.get(int(week), 0) for week in weeks])
        last_axis.bar(
            weeks,
            counts,
            bottom=bottom,
            width=0.86,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            label=label,
        )
        bottom += counts
    last_axis.set(
        title="Household theo tuần mua cuối cùng",
        xlabel="Tuần mua cuối cùng",
        ylabel="Số household",
    )
    last_axis.legend(frameon=False, fontsize=9, loc="upper left")

    values = np.sort(labelled["recency_days"].cast(pl.Float64).to_numpy())
    cumulative = np.arange(1, values.size + 1) / values.size
    recency_axis.plot(values, cumulative, color=BASE_DARK, linewidth=2.7)
    threshold_days = (7, 14, 30, 60, 90)
    threshold_shares = tuple(float((values <= days).mean()) for days in threshold_days)
    for days, share in zip(threshold_days, threshold_shares, strict=True):
        recency_axis.scatter(
            days,
            share,
            s=38,
            color=HIGHLIGHT,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
    for days in (30, 60, 90):
        recency_axis.axvline(
            days,
            color=REFERENCE,
            linewidth=1,
            linestyle="--",
            alpha=0.55,
        )
    median_recency = float(np.median(values))
    recency_axis.text(
        0.96,
        0.52,
        (
            f"≤ 7 ngày: {threshold_shares[0]:.1%}\n"
            f"≤ 14 ngày: {threshold_shares[1]:.1%}\n"
            f"≤ 30 ngày: {threshold_shares[2]:.1%}\n"
            f"≤ 60 ngày: {threshold_shares[3]:.1%}\n"
            f"≤ 90 ngày: {threshold_shares[4]:.1%}\n"
            f"Median: {median_recency:,.0f} ngày"
        ),
        transform=recency_axis.transAxes,
        ha="right",
        va="center",
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": "#D7DCE0",
        },
    )
    recency_axis.yaxis.set_major_formatter(PercentFormatter(1))
    recency_axis.set(
        title="Tỷ lệ household tích lũy theo recency",
        xlabel="Số ngày từ lần mua cuối đến snapshot",
        ylabel="Tỷ lệ household tích lũy",
        xlim=(0, max(100, float(values.max()) * 1.02)),
        ylim=(0, 1.02),
    )

    for axis in (last_axis, recency_axis):
        finish_axis(axis)
    figure.subplots_adjust(
        top=0.79,
        bottom=0.13,
        left=0.07,
        right=0.98,
        wspace=0.14,
    )
    figure.suptitle(
        "Last purchase và mức độ im lặng tại snapshot",
        fontsize=17,
        fontweight="bold",
        color=INK,
        y=0.97,
    )
    figure.text(
        0.5,
        0.91,
        (
            "Recency = snapshot DAY − DAY mua cuối; giá trị thấp cho biết "
            "household còn hoạt động gần snapshot"
        ),
        ha="center",
        color="#4B5563",
        fontsize=10.5,
    )
    return figure


def plot_household_interpurchase_dynamics(
    households: pl.DataFrame,
    event_distribution: pl.DataFrame,
) -> Figure:
    """Compare household cadence, regularity, and snapshot lateness."""
    require_columns(
        households,
        (
            "n_intervals",
            "median_interval_days",
            "interval_cv",
            "recency_ratio",
        ),
    )
    require_columns(
        event_distribution,
        ("return_interval_days", "cumulative_share"),
    )
    reliable = households.filter(
        (pl.col("n_intervals") >= 3)
        & pl.col("median_interval_days").is_not_null()
        & pl.col("interval_cv").is_not_null()
        & pl.col("recency_ratio").is_not_null()
    )
    median_intervals = reliable["median_interval_days"].cast(pl.Float64).to_numpy()
    interval_cv = reliable["interval_cv"].cast(pl.Float64).to_numpy()
    household_median = float(np.median(median_intervals))
    event_median = float(
        event_distribution.filter(pl.col("cumulative_share") >= 0.5)[
            "return_interval_days"
        ][0]
    )
    interval_cap = float(np.quantile(median_intervals, 0.99))

    figure = plt.figure(figsize=(15, 10.2), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(1.05, 1),
        hspace=0.42,
        wspace=0.20,
    )
    histogram_axis = figure.add_subplot(grid[0, :])
    regularity_axis = figure.add_subplot(grid[1, 0])
    lateness_axis = figure.add_subplot(grid[1, 1])

    central_intervals = median_intervals[median_intervals <= interval_cap]
    histogram_axis.hist(
        central_intervals,
        bins=np.arange(0.5, np.ceil(interval_cap) + 1.5, 1),
        color=BASE_DARK,
        edgecolor="white",
        linewidth=0.5,
    )
    histogram_axis.axvline(
        household_median,
        color=BASE,
        linewidth=3,
        linestyle="-",
        label=f"Median household = {household_median:g} ngày",
    )
    histogram_axis.axvline(
        event_median,
        color=HIGHLIGHT,
        linewidth=2.5,
        linestyle="--",
        label=f"Median event-level = {event_median:g} ngày",
    )
    histogram_axis.set(
        title="Phân bổ chu kỳ tái mua trung vị của từng household",
        xlabel="Median interpurchase interval / household (ngày, đến p99)",
        ylabel="Số household",
    )
    histogram_axis.legend(frameon=False, loc="upper right")

    cv_cap = float(np.quantile(interval_cv, 0.99))
    central_regularity = reliable.filter(pl.col("interval_cv") <= cv_cap)
    regularity_axis.hexbin(
        central_regularity["median_interval_days"].to_numpy(),
        central_regularity["interval_cv"].to_numpy(),
        gridsize=28,
        mincnt=1,
        bins="log",
        xscale="log",
        cmap=DENSITY_CMAP,
    )
    regularity_axis.axhline(
        1,
        color=HIGHLIGHT,
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
    )
    regularity_axis.set(
        title="Cadence điển hình × độ biến động",
        xlabel="Median interval (ngày, log scale)",
        ylabel="Interval CV (đến p99)",
        ylim=(0, cv_cap * 1.03),
    )
    regularity_axis.text(
        0.97,
        0.95,
        "CV thấp: mua đều hơn\nCV cao: mua thất thường hơn",
        transform=regularity_axis.transAxes,
        ha="right",
        va="top",
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#D7DCE0",
        },
    )

    lateness = reliable["recency_ratio"].cast(pl.Float64).to_numpy()
    band_counts = np.asarray(
        [
            int((lateness < 1).sum()),
            int(((lateness >= 1) & (lateness < 2)).sum()),
            int(((lateness >= 2) & (lateness < 4)).sum()),
            int((lateness >= 4).sum()),
        ]
    )
    band_shares = band_counts / band_counts.sum()
    band_masks = (
        lateness < 1,
        (lateness >= 1) & (lateness < 2),
        (lateness >= 2) & (lateness < 4),
        lateness >= 4,
    )
    band_median_intervals = tuple(
        float(np.median(median_intervals[mask])) if mask.any() else None
        for mask in band_masks
    )
    band_median_labels = tuple(
        "n/a" if value is None else f"{value:g} ngày"
        for value in band_median_intervals
    )
    band_labels = [
        f"{label}\nMedian: {median_label}"
        for (label, _color), median_label in zip(
            LATENESS_BANDS,
            band_median_labels,
            strict=True,
        )
    ]
    band_colors = [color for _label, color in LATENESS_BANDS]
    lateness_axis.bar(
        band_labels,
        band_shares,
        color=band_colors,
        edgecolor="white",
        width=0.74,
    )
    for index, share in enumerate(band_shares):
        lateness_axis.text(
            index,
            share + 0.015,
            f"{share:.1%} ({band_counts[index]:,})",
            ha="center",
            va="bottom",
            color=INK,
            fontsize=10,
        )
    lateness_axis.yaxis.set_major_formatter(PercentFormatter(1))
    lateness_axis.set(
        title="Household đang trễ bao nhiêu so với median interval riêng?",
        xlabel="Recency / median interval",
        ylabel="Tỷ lệ household",
        ylim=(0, max(0.55, float(band_shares.max()) + 0.10)),
    )
    lateness_axis.text(
        0.98,
        0.95,
        "1× = median interval của chính household\n"
        "Risk bands mô tả — không phải churn probability",
        transform=lateness_axis.transAxes,
        ha="right",
        va="top",
        color="#4B5563",
        fontsize=9.5,
    )

    for axis in (histogram_axis, regularity_axis, lateness_axis):
        finish_axis(axis)
    figure.subplots_adjust(
        top=0.84,
        bottom=0.09,
        left=0.07,
        right=0.98,
    )
    figure.suptitle(
        "Nhịp mua tự nhiên, độ đều và mức độ trễ của household",
        fontsize=18,
        fontweight="bold",
        color=INK,
        y=0.97,
    )
    figure.text(
        0.5,
        0.925,
        (
            "Mỗi household đóng góp đúng một median interval; chỉ household "
            "có ít nhất 3 khoảng mua được dùng cho regularity"
        ),
        ha="center",
        color="#4B5563",
        fontsize=10.5,
    )
    return figure

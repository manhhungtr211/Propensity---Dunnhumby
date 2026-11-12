"""Population and archetype views of recent household behavior shift."""

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

STATE_STYLE: Final = (
    ("Accelerating", "Tăng tốc", BASE_DARK),
    ("Stable", "Ổn định", BASE),
    ("Cooling", "Giảm nhịp", "#E9A85B"),
    ("Dormant", "Không mua gần đây", HIGHLIGHT),
    ("Newly active", "Mới hoạt động", REFERENCE),
)


def plot_household_behavior_shift(households: pl.DataFrame) -> Figure:
    """Show full-population change before compressed archetype summaries."""
    require_columns(
        households,
        (
            "cutoff_day",
            "history_start_day",
            "history_end_day",
            "recent_start_day",
            "historical_basket_rate_30d",
            "recent_basket_rate_30d",
            "median_products_per_basket_historical",
            "median_products_per_basket_recent",
            "frequency_state",
        ),
    )
    historical_rate = households["historical_basket_rate_30d"].to_numpy()
    recent_rate = households["recent_basket_rate_30d"].to_numpy()
    joint_cap = float(
        max(
            np.quantile(historical_rate, 0.99),
            np.quantile(recent_rate, 0.99),
        )
    )
    central = households.filter(
        (pl.col("historical_basket_rate_30d") <= joint_cap)
        & (pl.col("recent_basket_rate_30d") <= joint_cap)
    )

    figure = plt.figure(figsize=(15, 10.2), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(1.05, 1),
        hspace=0.42,
        wspace=0.22,
    )
    population_axis = figure.add_subplot(grid[0, :])
    state_axis = figure.add_subplot(grid[1, 0])
    slope_axis = figure.add_subplot(grid[1, 1])

    population_axis.hexbin(
        central["historical_basket_rate_30d"].to_numpy(),
        central["recent_basket_rate_30d"].to_numpy(),
        gridsize=38,
        mincnt=1,
        bins="log",
        cmap=DENSITY_CMAP,
    )
    population_axis.plot(
        [0, joint_cap],
        [0, joint_cap],
        color=HIGHLIGHT,
        linewidth=2,
        linestyle="--",
        label="Không đổi (Y = X)",
    )
    population_axis.set(
        title="Toàn bộ household: purchase rate lịch sử vs gần đây",
        xlabel="Historical baskets / 30 ngày tương đương",
        ylabel="Recent baskets / 30 ngày",
        xlim=(0, joint_cap),
        ylim=(0, joint_cap),
    )
    population_axis.legend(frameon=False, loc="upper left")
    population_axis.text(
        0.98,
        0.05,
        (
            "Trên đường chéo: tăng tốc\n"
            "Dưới đường chéo: giảm nhịp\n"
            "Scatter hiển thị đến joint p99"
        ),
        transform=population_axis.transAxes,
        ha="right",
        va="bottom",
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#D7DCE0",
        },
    )

    counts_by_state = dict(
        households.group_by("frequency_state")
        .agg(pl.len().alias("n_households"))
        .iter_rows()
    )
    state_counts = np.asarray(
        [counts_by_state.get(state, 0) for state, _label, _color in STATE_STYLE]
    )
    state_shares = state_counts / state_counts.sum()
    state_axis.bar(
        [label for _state, label, _color in STATE_STYLE],
        state_shares,
        color=[color for _state, _label, color in STATE_STYLE],
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
    state_axis.tick_params(axis="x", rotation=12)
    state_axis.yaxis.set_major_formatter(PercentFormatter(1))
    state_axis.set(
        title="Nén population thành frequency transitions",
        ylabel="Tỷ lệ household",
        ylim=(0, max(0.55, float(state_shares.max()) + 0.10)),
    )

    active_states = ("Accelerating", "Stable", "Cooling")
    representative = (
        households.filter(
            pl.col("frequency_state").is_in(active_states)
            & pl.col("median_products_per_basket_historical").is_not_null()
            & pl.col("median_products_per_basket_recent").is_not_null()
        )
        .group_by("frequency_state")
        .agg(
            pl.col("median_products_per_basket_historical")
            .median()
            .alias("historical"),
            pl.col("median_products_per_basket_recent")
            .median()
            .alias("recent"),
        )
    )
    style_by_state = {
        state: (label, color) for state, label, color in STATE_STYLE
    }
    for row in representative.iter_rows(named=True):
        state = str(row["frequency_state"])
        label, color = style_by_state[state]
        historical_value = float(row["historical"])
        recent_value = float(row["recent"])
        slope_axis.plot(
            [0, 1],
            [historical_value, recent_value],
            color=color,
            linewidth=3,
            marker="o",
            markersize=7,
            label=label,
        )
        slope_axis.text(
            -0.04,
            historical_value,
            f"{historical_value:g}",
            ha="right",
            va="center",
            color=color,
        )
        slope_axis.text(
            1.04,
            recent_value,
            f"{recent_value:g}",
            ha="left",
            va="center",
            color=color,
        )
    slope_axis.set_xticks((0, 1), ("Historical", "Recent 30 ngày"))
    slope_axis.set_xlim(-0.15, 1.15)
    slope_axis.set(
        title="Đại diện nhóm: median products / basket",
        ylabel="Số product distinct / basket",
    )
    slope_axis.legend(frameon=False, loc="best")

    for axis in (population_axis, state_axis, slope_axis):
        finish_axis(axis)
    cutoff = int(households["cutoff_day"][0])
    history_start = int(households["history_start_day"][0])
    history_end = int(households["history_end_day"][0])
    recent_start = int(households["recent_start_day"][0])
    figure.subplots_adjust(
        top=0.84,
        bottom=0.10,
        left=0.08,
        right=0.98,
    )
    figure.suptitle(
        "Long-term baseline → recent behavior shift",
        fontsize=18,
        fontweight="bold",
        color=INK,
        y=0.97,
    )
    figure.text(
        0.5,
        0.925,
        (
            f"Historical DAY {history_start}–{history_end} (365 ngày) vs "
            f"recent DAY {recent_start}–{cutoff} (30 ngày), không chồng lấn"
        ),
        ha="center",
        color="#4B5563",
        fontsize=10.5,
    )
    return figure

"""Transaction activity summaries and the overview dashboard figure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

MONTHS_PER_YEAR: Final = 12
DAYS_PER_RELATIVE_YEAR: Final = 365
DAYS_PER_MONTH: Final = 30.44
TIME_SEGMENT_ORDER: Final[tuple[str, ...]] = (
    "Sáng\n05–10h",
    "Trưa\n11–13h",
    "Chiều\n14–17h",
    "Tối\n18–04h",
)


class ActivityDashboardError(ValueError):
    """Raised when transaction data violates the dashboard contract."""


@dataclass(frozen=True, slots=True)
class TransactionActivityDashboard:
    """Prepared summaries for the three-panel activity dashboard."""

    monthly_baskets: pd.DataFrame
    time_segment_baskets: pd.DataFrame
    household_history: pd.DataFrame


def load_transaction_activity_dashboard(
    transaction_path: Path,
) -> TransactionActivityDashboard:
    """Load transaction columns and prepare dashboard summaries."""
    transactions = pd.read_csv(
        transaction_path,
        usecols=["household_key", "BASKET_ID", "DAY", "TRANS_TIME"],
    )
    return build_transaction_activity_dashboard(transactions)


def build_transaction_activity_dashboard(
    transactions: pd.DataFrame,
) -> TransactionActivityDashboard:
    """Build monthly, time-of-day, and household-history summaries."""
    required = {"household_key", "BASKET_ID", "DAY", "TRANS_TIME"}
    if missing := required.difference(transactions.columns):
        raise ActivityDashboardError(
            f"Transaction columns are missing: {sorted(missing)}"
        )
    basket_rows = (
        transactions[["household_key", "BASKET_ID", "DAY", "TRANS_TIME"]]
        .dropna()
        .drop_duplicates("BASKET_ID")
        .copy()
    )
    if basket_rows.empty:
        raise ActivityDashboardError("No valid baskets are available.")

    ordinal_day = basket_rows["DAY"].astype("int64")
    basket_rows["relative_year"] = (
        (ordinal_day - 1) // DAYS_PER_RELATIVE_YEAR + 1
    )
    day_in_year = (ordinal_day - 1) % DAYS_PER_RELATIVE_YEAR
    basket_rows["relative_month"] = (
        day_in_year * MONTHS_PER_YEAR // DAYS_PER_RELATIVE_YEAR + 1
    )
    monthly = (
        basket_rows.loc[basket_rows["relative_year"].isin([1, 2])]
        .groupby(["relative_year", "relative_month"], as_index=False)
        .agg(n_baskets=("BASKET_ID", "nunique"))
    )
    month_grid = pd.MultiIndex.from_product(
        [[1, 2], range(1, MONTHS_PER_YEAR + 1)],
        names=["relative_year", "relative_month"],
    )
    monthly = (
        monthly.set_index(["relative_year", "relative_month"])
        .reindex(month_grid, fill_value=0)
        .reset_index()
    )

    transaction_time = basket_rows["TRANS_TIME"].astype("int64")
    hour = transaction_time // 100
    minute = transaction_time % 100
    valid_clock = hour.between(0, 23) & minute.between(0, 59)
    clock_rows = basket_rows.loc[valid_clock].copy()
    clock_hour = hour.loc[valid_clock]
    clock_rows["time_segment"] = "Tối\n18–04h"
    clock_rows.loc[clock_hour.between(5, 10), "time_segment"] = "Sáng\n05–10h"
    clock_rows.loc[clock_hour.between(11, 13), "time_segment"] = "Trưa\n11–13h"
    clock_rows.loc[clock_hour.between(14, 17), "time_segment"] = "Chiều\n14–17h"
    time_segment = (
        clock_rows.groupby("time_segment", as_index=False)
        .agg(n_baskets=("BASKET_ID", "nunique"))
        .set_index("time_segment")
        .reindex(TIME_SEGMENT_ORDER, fill_value=0)
        .rename_axis("time_segment")
        .reset_index()
    )

    household_history = basket_rows.groupby("household_key", as_index=False).agg(
        first_day=("DAY", "min"),
        last_day=("DAY", "max"),
    )
    household_history["history_days"] = (
        household_history["last_day"] - household_history["first_day"] + 1
    )
    household_history["history_months"] = (
        household_history["history_days"] / DAYS_PER_MONTH
    )
    return TransactionActivityDashboard(monthly, time_segment, household_history)


def plot_transaction_activity_dashboard(
    dashboard: TransactionActivityDashboard,
) -> Figure:
    """Plot monthly trend, time segments, and household-history distribution."""
    figure = plt.figure(figsize=(16, 10), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=[1, 1.2])
    trend_axis = figure.add_subplot(grid[0, 0])
    time_axis = figure.add_subplot(grid[0, 1])
    history_axis = figure.add_subplot(grid[1, :])

    for relative_year, color in ((1, "#2563EB"), (2, "#F97316")):
        year_data = dashboard.monthly_baskets.loc[
            dashboard.monthly_baskets["relative_year"] == relative_year
        ]
        trend_axis.plot(
            year_data["relative_month"],
            year_data["n_baskets"],
            marker="o",
            linewidth=2.2,
            color=color,
            label=f"Năm {relative_year}",
        )
    trend_axis.set_title("Lượng giao dịch theo tháng: Năm 1 và Năm 2")
    trend_axis.set_xlabel("Tháng tương đối")
    trend_axis.set_ylabel("Số basket unique")
    trend_axis.set_xticks(range(1, MONTHS_PER_YEAR + 1))
    trend_axis.legend(frameon=False)
    trend_axis.spines[["top", "right"]].set_visible(False)

    time_bars = time_axis.bar(
        dashboard.time_segment_baskets["time_segment"],
        dashboard.time_segment_baskets["n_baskets"],
        color=["#93C5FD", "#60A5FA", "#2563EB", "#1E3A8A"],
    )
    time_axis.bar_label(time_bars, fmt="{:,.0f}", padding=4, fontsize=9)
    time_axis.set_title("Phân bố giao dịch theo khung giờ")
    time_axis.set_xlabel("")
    time_axis.set_ylabel("Số basket unique")
    time_axis.spines[["top", "right"]].set_visible(False)

    history_days = dashboard.household_history["history_days"].to_numpy()
    maximum_day = int(np.ceil(history_days.max() / 30) * 30)
    bin_edges = np.arange(0, maximum_day + 31, 30)
    counts, edges = np.histogram(history_days, bins=bin_edges)
    colors = ["#D1D5DB" if left < 90 else "#F97316" for left in edges[:-1]]
    history_axis.bar(
        edges[:-1],
        counts,
        width=np.diff(edges),
        align="edge",
        color=colors,
        edgecolor="white",
    )
    history_axis.axvline(90, color="#0F172A", linestyle="--", linewidth=1.8)
    history_axis.annotate(
        "Mốc tối thiểu 90 ngày",
        xy=(90, max(counts) * 0.82),
        xytext=(135, max(counts) * 0.95),
        arrowprops={"arrowstyle": "->", "color": "#0F172A"},
        color="#0F172A",
        fontsize=10,
    )
    history_axis.set_title("Độ dài lịch sử hoạt động của household")
    history_axis.set_xlabel("Độ dài lịch sử (ngày)")
    history_axis.set_ylabel("Số household")
    history_axis.spines[["top", "right"]].set_visible(False)
    secondary_axis = history_axis.secondary_xaxis(
        "top",
        functions=(
            lambda days: days / DAYS_PER_MONTH,
            lambda months: months * DAYS_PER_MONTH,
        ),
    )
    secondary_axis.set_xlabel("Độ dài lịch sử xấp xỉ (tháng)")
    figure.suptitle("Tổng quan nhịp giao dịch và độ sâu lịch sử", fontsize=16)
    return figure

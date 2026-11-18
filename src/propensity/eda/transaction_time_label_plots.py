"""Box plots for transaction-time features by binary label."""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from propensity.eda.transaction_time_label import (
    FEATURE_COLUMNS,
    FEATURE_TITLES,
    FOCUS_FEATURE_COLUMNS,
)

LABEL_PALETTE: Final = {0: "#64748B", 1: "#F97316"}
FOCUS_YLABELS: Final = {
    "days_since_last_purchase": "Số ngày từ lần mua cuối đến cutoff",
    "active_days_90d": "Số active days trong 90 ngày",
    "interpurchase_gap_std_days_90d": "Độ lệch chuẩn gap (ngày)",
}


def plot_transaction_time_boxplots(feature_panel: pd.DataFrame) -> Figure:
    """Plot one box plot per numerical temporal feature on its natural scale."""
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
        layout="constrained",
    )
    for axis, feature in zip(axes.flat, FEATURE_COLUMNS, strict=False):
        plot_data = feature_panel[["label", feature]].dropna()
        sns.boxplot(
            data=plot_data,
            x="label",
            y=feature,
            order=[0, 1],
            hue="label",
            hue_order=[0, 1],
            palette=LABEL_PALETTE,
            legend=False,
            whis=1.5,
            showfliers=True,
            flierprops={
                "marker": "o",
                "markersize": 2,
                "markerfacecolor": "#334155",
                "markeredgecolor": "none",
                "alpha": 0.18,
            },
            ax=axis,
        )
        axis.set_title(FEATURE_TITLES[feature])
        axis.set_xlabel("Label (0 = Negative, 1 = Positive)")
        axis.set_ylabel("Ngày" if "days" in feature else "Số lượng")
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)

    for axis in axes.flat[len(FEATURE_COLUMNS) :]:
        axis.set_visible(False)
    figure.suptitle("Transaction-time features theo label", fontsize=15)
    return figure


def plot_transaction_time_focus_boxplots(feature_panel: pd.DataFrame) -> Figure:
    """Plot three focused box plots while hiding outlier markers."""
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )
    for axis, feature in zip(axes, FOCUS_FEATURE_COLUMNS, strict=True):
        plot_data = feature_panel[["label", feature]].dropna()
        sns.boxplot(
            data=plot_data,
            x="label",
            y=feature,
            order=[0, 1],
            hue="label",
            hue_order=[0, 1],
            palette=LABEL_PALETTE,
            legend=False,
            whis=1.5,
            showfliers=False,
            ax=axis,
        )
        axis.set_title(FEATURE_TITLES[feature])
        axis.set_xlabel("Label (0 = Negative, 1 = Positive)")
        axis.set_ylabel(FOCUS_YLABELS[feature])
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Focused transaction-time distributions", fontsize=15)
    figure.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.23, wspace=0.30)
    figure.text(
        0.5,
        0.05,
        f"Tập dữ liệu N = {len(feature_panel)}. Biểu đồ ẩn các điểm ngoại lai "
        "(chiếm khoảng 3–10% mẫu ở mỗi biến).",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#475569",
    )
    return figure

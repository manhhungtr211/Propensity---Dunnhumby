"""Box plots for quantity and basket-diversity features by label."""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from propensity.eda.transaction_quantity_basket_label import (
    FEATURE_COLUMNS,
    FEATURE_TITLES,
    FEATURE_Y_LABELS,
)

LABEL_PALETTE: Final = {0: "#64748B", 1: "#F97316"}


def plot_quantity_basket_boxplots(feature_panel: pd.DataFrame) -> Figure:
    """Plot quantity and basket-diversity distributions for both labels."""
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
        layout="constrained",
    )
    for axis, feature in zip(axes.flat, FEATURE_COLUMNS, strict=True):
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
        axis.set_ylabel(FEATURE_Y_LABELS[feature])
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "Transaction quantity và basket diversity theo label",
        fontsize=15,
    )
    return figure


def plot_department_breadth_boxplot(feature_panel: pd.DataFrame) -> Figure:
    """Compare 90-day department breadth between Negative and Positive labels."""
    figure, axis = plt.subplots(figsize=(7, 5), layout="constrained")
    sns.boxplot(
        data=feature_panel,
        x="label",
        y="n_departments_90d",
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
    axis.set_title("Số department unique đã mua trong 90 ngày")
    axis.set_xlabel("Label (0 = Negative, 1 = Positive)")
    axis.set_ylabel("Số department unique")
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)
    return figure

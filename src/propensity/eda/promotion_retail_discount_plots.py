"""Plots for retail-discount features by purchase-propensity label."""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from propensity.eda.promotion_retail_discount_label import (
    BOXPLOT_FEATURES,
    FEATURE_TITLES,
)

LABEL_PALETTE: Final = {0: "#64748B", 1: "#F97316"}


def plot_retail_discount_ever_positive_rate(feature_panel: pd.DataFrame) -> Figure:
    """Plot the Positive rate for households with and without retail discount."""
    summary = (
        feature_panel.groupby("ever_retail_discount_90d", as_index=False)
        .agg(positive_rate=("label", "mean"), n_observations=("label", "size"))
        .sort_values("ever_retail_discount_90d")
    )
    summary["segment"] = summary["ever_retail_discount_90d"].map(
        {0: "Chưa từng có retail discount", 1: "Từng có retail discount"}
    )
    figure, axis = plt.subplots(figsize=(10, 4.5), layout="constrained")
    bars = axis.barh(summary["segment"], summary["positive_rate"], color="#F97316")
    axis.bar_label(
        bars,
        labels=[
            f"{rate:.1%}  (n={count:,})"
            for rate, count in zip(
                summary["positive_rate"],
                summary["n_observations"],
                strict=True,
            )
        ],
        padding=5,
        fontsize=9,
    )
    axis.set_title("Positive rate theo lịch sử sử dụng retail discount")
    axis.set_xlabel("Tỷ lệ label Positive")
    axis.set_ylabel("")
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.grid(axis="y", visible=False)
    axis.spines[["top", "right", "left"]].set_visible(False)
    return figure


def plot_retail_discount_boxplots(feature_panel: pd.DataFrame) -> Figure:
    """Compare continuous retail-discount features across both labels."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for axis, feature in zip(axes, BOXPLOT_FEATURES, strict=True):
        sns.boxplot(
            data=feature_panel,
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
        is_ratio = "share" in feature or "to_sales" in feature
        axis.set_ylabel("Tỷ lệ" if is_ratio else "Discount")
        if is_ratio:
            axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Retail-discount features: P25–P75 theo label",
        fontsize=15,
    )
    return figure

"""Plots for historical coupon features by propensity label."""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from propensity.eda.coupon_label import (
    BOXPLOT_FEATURES,
    FEATURE_TITLES,
    FEATURE_Y_LABELS,
)

LABEL_PALETTE: Final = {0: "#64748B", 1: "#F97316"}
COUPON_USAGE_PALETTE: Final = {0: "#CBD5E1", 1: "#F97316"}


def plot_coupon_user_positive_rate(feature_panel: pd.DataFrame) -> Figure:
    """Plot Positive rates for coupon users and non-users."""
    summary = (
        feature_panel.groupby("ever_coupon_redeemed_90d", as_index=False)
        .agg(positive_rate=("label", "mean"), n_observations=("label", "size"))
        .sort_values("ever_coupon_redeemed_90d")
    )
    summary["segment"] = summary["ever_coupon_redeemed_90d"].map(
        {0: "Chưa từng redeem coupon", 1: "Từng redeem coupon"}
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
    axis.set_title("Positive rate theo lịch sử coupon redemption")
    axis.set_xlabel("Tỷ lệ label Positive")
    axis.set_ylabel("")
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.grid(axis="y", visible=False)
    axis.spines[["top", "right", "left"]].set_visible(False)
    return figure


def plot_coupon_usage_by_label_stacked(feature_panel: pd.DataFrame) -> Figure:
    """Plot the used/not-used coupon percentage composition inside each label."""
    shares = pd.crosstab(
        feature_panel["label"],
        feature_panel["ever_coupon_redeemed_90d"],
        normalize="index",
    ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    figure, axis = plt.subplots(figsize=(10, 4.5), layout="constrained")
    unused_bars = axis.barh(
        ["Negative (0)", "Positive (1)"],
        shares[0],
        color=COUPON_USAGE_PALETTE[0],
        label="Chưa redeem coupon",
    )
    used_bars = axis.barh(
        ["Negative (0)", "Positive (1)"],
        shares[1],
        left=shares[0],
        color=COUPON_USAGE_PALETTE[1],
        label="Đã redeem coupon",
    )
    for bars, values in ((unused_bars, shares[0]), (used_bars, shares[1])):
        axis.bar_label(
            bars,
            labels=[f"{value:.1%}" for value in values],
            label_type="center",
            color="#0F172A",
            fontsize=9,
        )
    axis.set_title("Tỷ lệ sử dụng coupon trong từng label")
    axis.set_xlabel("Tỷ lệ household-cutoff")
    axis.set_ylabel("")
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.legend(title="Coupon usage", frameon=False, ncol=2, loc="lower right")
    axis.grid(axis="y", visible=False)
    axis.spines[["top", "right", "left"]].set_visible(False)
    return figure


def plot_coupon_feature_boxplots(feature_panel: pd.DataFrame) -> Figure:
    """Compare coupon counts, recency, and discount totals by label."""
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for axis, feature in zip(axes.flat, BOXPLOT_FEATURES, strict=True):
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
        axis.set_ylabel(FEATURE_Y_LABELS[feature])
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Coupon features: P25–P75 theo label", fontsize=15)
    return figure

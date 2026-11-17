"""Plots for demographic distributions conditioned on binary labels."""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from propensity.eda.demographic_label_config import (
    CATEGORY_ORDERS,
    FEATURES,
    MISSING_GROUP,
)

BAR_COLOR: Final = "#2563EB"
MISSING_COLOR: Final = "#9CA3AF"
BASELINE_COLOR: Final = "#DC2626"
LABEL_PALETTE: Final = {"Negative": "#64748B", "Positive": "#F97316"}


def plot_label_distributions(
    distribution_summary: pd.DataFrame,
) -> tuple[Figure, ...]:
    """Plot normalized category shares within Positive and Negative labels."""
    figures: list[Figure] = []
    for feature in FEATURES:
        plot_data = distribution_summary.loc[
            distribution_summary["feature"] == feature
        ].copy()
        observed = plot_data["category"].astype(str).unique().tolist()
        preferred = CATEGORY_ORDERS[feature]
        extras = sorted(set(observed).difference(preferred, {MISSING_GROUP}))
        category_order = [value for value in preferred if value in observed] + extras
        if MISSING_GROUP in observed:
            category_order.append(MISSING_GROUP)

        figure, axis = plt.subplots(
            figsize=(10, max(4, 0.55 * len(category_order))),
            layout="constrained",
        )
        sns.barplot(
            data=plot_data,
            y="category",
            x="within_label_share",
            hue="label_name",
            order=category_order,
            hue_order=["Negative", "Positive"],
            palette=LABEL_PALETTE,
            ax=axis,
        )
        axis.set_title(f"Phân phối {feature} trong Positive và Negative")
        axis.set_xlabel("Tỷ trọng trong từng label")
        axis.set_ylabel("")
        axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        axis.legend(title="Label", frameon=False)
        figures.append(figure)
    return tuple(figures)


def plot_positive_rate_by_category(
    positive_rate_summary: pd.DataFrame,
    overall_positive_rate: float,
    include_missing: bool = False,
) -> tuple[Figure, ...]:
    """Plot one decluttered horizontal positive-rate chart per feature."""
    figures: list[Figure] = []
    for feature in FEATURES:
        plot_data = positive_rate_summary.loc[
            positive_rate_summary["feature"] == feature
        ].copy()
        if not include_missing:
            plot_data = plot_data.loc[plot_data["category"] != MISSING_GROUP]
        plot_data = plot_data.sort_values("positive_rate")
        colors = [
            MISSING_COLOR if value else BAR_COLOR
            for value in plot_data["is_missing_category"]
        ]

        figure, axis = plt.subplots(
            figsize=(10, max(4, 0.55 * len(plot_data))),
            layout="constrained",
        )
        bars = axis.barh(
            plot_data["category"],
            plot_data["positive_rate"],
            color=colors,
        )
        axis.axvline(
            overall_positive_rate,
            color=BASELINE_COLOR,
            linestyle="--",
            linewidth=1.2,
            label=f"Toàn bộ: {overall_positive_rate:.1%}",
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.1%}" for value in plot_data["positive_rate"]],
            padding=4,
        )
        upper_limit = max(
            float(plot_data["positive_rate"].max()), overall_positive_rate
        )
        axis.set_xlim(0, min(1.0, upper_limit * 1.18))
        axis.set_title(f"Tỷ lệ label dương theo {feature}")
        axis.set_xlabel("Positive rate: tỷ lệ label = 1")
        axis.set_ylabel("")
        axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        axis.legend(frameon=False, loc="lower right")
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.grid(axis="y", visible=False)
        figures.append(figure)
    return tuple(figures)

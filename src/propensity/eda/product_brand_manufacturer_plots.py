"""Brand and manufacturer distribution plots by label."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

LABEL_PALETTE: Final = {"Negative": "#64748B", "Positive": "#F97316"}
BOX_PALETTE: Final = {0: "#64748B", 1: "#F97316"}


@dataclass(frozen=True, slots=True)
class ManufacturerPlotConfig:
    """Support and display limits for manufacturer plots."""

    top_n: int = 20
    min_observations: int = 300


def plot_brand_history_distribution(summary: pd.DataFrame) -> Figure:
    """Compare National/Private purchase history across both labels."""
    category_order = _negative_count_order(summary)
    return _plot_grouped_bars(
        summary,
        category_order,
        "Historical brand: Negative vs Positive",
        "Tỷ lệ household-cutoff từng mua brand",
    )


def plot_dominant_brand_distribution(summary: pd.DataFrame) -> Figure:
    """Compare dominant-brand composition across both labels."""
    category_order = _negative_count_order(summary)
    return _plot_grouped_bars(
        summary,
        category_order,
        "Dominant brand: Negative vs Positive",
        "Tỷ trọng trong từng label",
    )


def plot_manufacturer_history_distribution(
    summary: pd.DataFrame,
    config: ManufacturerPlotConfig | None = None,
) -> Figure:
    """Plot supported manufacturers with the largest label-distribution gaps."""
    plot_config = config or ManufacturerPlotConfig()
    metrics = summary[
        ["category", "total_observations", "absolute_gap_pp"]
    ].drop_duplicates()
    selected = metrics.loc[
        metrics["total_observations"] >= plot_config.min_observations
    ].nlargest(plot_config.top_n, "absolute_gap_pp")
    plot_data = summary.loc[summary["category"].isin(selected["category"])].copy()
    category_order = _negative_count_order(plot_data)
    return _plot_grouped_bars(
        plot_data,
        category_order,
        "Historical manufacturer: Negative vs Positive",
        "Tỷ lệ household-cutoff từng mua manufacturer",
    )


def plot_brand_diversity_boxplots(feature_panel: pd.DataFrame) -> Figure:
    """Compare brand breadth and dominant-brand concentration by label."""
    features = (
        "n_brands_90d",
        "dominant_brand_share_90d",
    )
    titles = {
        "n_brands_90d": "Số brand đã mua",
        "dominant_brand_share_90d": "Dominant-brand share",
    }
    y_labels = {
        "n_brands_90d": "Số brand unique trong 90 ngày",
        "dominant_brand_share_90d": "Tỷ trọng product lines của dominant brand",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for axis, feature in zip(axes, features, strict=True):
        plot_data = feature_panel[["label", feature]].dropna()
        sns.boxplot(
            data=plot_data,
            x="label",
            y=feature,
            order=[0, 1],
            hue="label",
            hue_order=[0, 1],
            palette=BOX_PALETTE,
            legend=False,
            whis=(5, 95),
            showfliers=False,
            ax=axis,
        )
        axis.set_title(titles[feature])
        axis.set_xlabel("Label (0 = Negative, 1 = Positive)")
        axis.set_ylabel(y_labels[feature])
        if feature == "dominant_brand_share_90d":
            axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        axis.spines[["top", "right"]].set_visible(False)
    return figure


def _negative_count_order(summary: pd.DataFrame) -> list[str]:
    negative = summary.loc[
        summary["label"] == 0,
        ["category", "n_observations"],
    ].sort_values("n_observations")
    return negative["category"].astype(str).tolist()


def _plot_grouped_bars(
    summary: pd.DataFrame,
    category_order: list[str],
    title: str,
    x_label: str,
) -> Figure:
    prepared = summary.assign(
        category=summary["category"].astype(str),
        label_name=summary["label"].map({0: "Negative", 1: "Positive"}),
    )
    figure, axis = plt.subplots(
        figsize=(11, max(4, 0.5 * len(category_order))),
        layout="constrained",
    )
    sns.barplot(
        data=prepared,
        y="category",
        x="within_label_share",
        hue="label_name",
        order=category_order,
        hue_order=["Negative", "Positive"],
        palette=LABEL_PALETTE,
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("")
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.legend(title="Label", frameon=False)
    axis.grid(axis="y", visible=False)
    axis.spines[["top", "right", "left"]].set_visible(False)
    return figure

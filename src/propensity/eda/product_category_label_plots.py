"""Plots for historical product-category distributions by label."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from propensity.eda.product_category_label import BREADTH_FEATURES

LABEL_PALETTE: Final = {"Negative": "#64748B", "Positive": "#F97316"}
BREADTH_PALETTE: Final = {0: "#64748B", 1: "#F97316"}


@dataclass(frozen=True, slots=True)
class CategoryPlotConfig:
    """Support and display limits for high-cardinality category plots."""

    top_n: int = 20
    min_observations: int = 300


def plot_historical_category_distributions(
    summary: pd.DataFrame,
    config: CategoryPlotConfig | None = None,
) -> tuple[Figure, ...]:
    """Compare historical-category prevalence in Negative and Positive rows."""
    plot_config = config or CategoryPlotConfig()
    figures: list[Figure] = []
    for dimension in ("department", "commodity", "subcommodity"):
        dimension_data = summary.loc[summary["dimension"] == dimension]
        category_metrics = dimension_data[
            [
                "category",
                "total_observations",
                "positive_minus_negative_pp",
                "absolute_gap_pp",
            ]
        ].drop_duplicates()
        selected = category_metrics.loc[
            category_metrics["total_observations"] >= plot_config.min_observations
        ].nlargest(plot_config.top_n, "absolute_gap_pp")
        negative_counts = dimension_data.loc[
            dimension_data["label"] == 0,
            ["category", "n_observations"],
        ].rename(columns={"n_observations": "negative_observations"})
        selected = selected.merge(
            negative_counts,
            on="category",
            how="left",
            validate="one_to_one",
        ).fillna({"negative_observations": 0})
        category_order = selected.sort_values("negative_observations")[
            "category"
        ].tolist()
        plot_data = dimension_data.loc[
            dimension_data["category"].isin(category_order)
        ].copy()
        figures.append(
            _plot_grouped_distribution(
                plot_data,
                category_order,
                f"Historical {dimension}: Negative vs Positive",
                "Tỷ lệ household-cutoff từng mua category",
            )
        )
    return tuple(figures)


def plot_dominant_category_distributions(
    summary: pd.DataFrame,
    config: CategoryPlotConfig | None = None,
) -> tuple[Figure, ...]:
    """Compare dominant-category composition inside each label."""
    plot_config = config or CategoryPlotConfig()
    figures: list[Figure] = []
    for dimension in ("department", "commodity"):
        dimension_data = summary.loc[summary["dimension"] == dimension]
        category_metrics = dimension_data[
            [
                "category",
                "total_observations",
                "positive_minus_negative_pp",
            ]
        ].drop_duplicates()
        selected = category_metrics.loc[
            category_metrics["total_observations"] >= plot_config.min_observations
        ].nlargest(plot_config.top_n, "total_observations")
        negative_counts = dimension_data.loc[
            dimension_data["label"] == 0,
            ["category", "n_observations"],
        ].rename(columns={"n_observations": "negative_observations"})
        selected = selected.merge(
            negative_counts,
            on="category",
            how="left",
            validate="one_to_one",
        ).fillna({"negative_observations": 0})
        category_order = selected.sort_values("negative_observations")[
            "category"
        ].tolist()
        plot_data = dimension_data.loc[
            dimension_data["category"].isin(category_order)
        ].copy()
        figures.append(
            _plot_grouped_distribution(
                plot_data,
                category_order,
                f"Dominant {dimension}: Negative vs Positive",
                "Tỷ trọng trong từng label",
            )
        )
    return tuple(figures)


def plot_category_breadth_boxplots(feature_panel: pd.DataFrame) -> Figure:
    """Compare department and commodity breadth across both labels."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    titles = {
        "n_departments_90d": "Số department đã mua",
        "n_commodities_90d": "Số commodity đã mua",
    }
    for axis, feature in zip(axes, BREADTH_FEATURES, strict=True):
        sns.boxplot(
            data=feature_panel,
            x="label",
            y=feature,
            order=[0, 1],
            hue="label",
            hue_order=[0, 1],
            palette=BREADTH_PALETTE,
            legend=False,
            whis=(5, 95),
            showfliers=False,
            ax=axis,
        )
        axis.set_title(titles[feature])
        axis.set_xlabel("Label (0 = Negative, 1 = Positive)")
        axis.set_ylabel("Số category unique trong 90 ngày")
        axis.spines[["top", "right"]].set_visible(False)
    return figure


def _plot_grouped_distribution(
    plot_data: pd.DataFrame,
    category_order: list[str],
    title: str,
    x_label: str,
) -> Figure:
    prepared = plot_data.assign(
        label_name=plot_data["label"].map({0: "Negative", 1: "Positive"})
    )
    figure, axis = plt.subplots(
        figsize=(11, max(5, 0.48 * len(category_order))),
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

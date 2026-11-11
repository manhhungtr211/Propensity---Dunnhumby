"""Store, discount, cross-behavior, and pattern views."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from propensity.eda.visualization.transaction_behavior_common import (
    BLUE,
    GRAY,
    GREEN,
    ORANGE,
    PURPLE,
    finish_axis,
    numeric_values,
    require_columns,
)


def plot_store_usage(households: pl.DataFrame) -> Figure:
    """Show store breadth and concentration while retaining frequency context."""
    require_columns(
        households,
        ("unique_stores", "primary_store_share", "basket_count"),
    )
    stores = numeric_values(households, "unique_stores")
    share = numeric_values(households, "primary_store_share")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    axes[0].hist(
        stores,
        bins=np.arange(0.5, stores.max() + 1.5),
        color=GREEN,
        rwidth=0.9,
    )
    axes[0].set(
        title="Store breadth", xlabel="Số store distinct", ylabel="Số household"
    )
    axes[1].hist(share, bins=np.linspace(0, 1, 26), color=PURPLE)
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set(
        title="Mức tập trung tại primary store",
        xlabel="Primary-store share",
        ylabel="Số household",
    )
    axes[2].scatter(
        households["basket_count"].to_numpy(),
        households["primary_store_share"].to_numpy(),
        s=14,
        alpha=0.35,
        color=BLUE,
        edgecolors="none",
    )
    axes[2].set_xscale("log")
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].set(
        title="Concentration trong bối cảnh frequency",
        xlabel="Basket count (log)",
        ylabel="Primary-store share",
    )
    for axis in axes:
        finish_axis(axis)
    figure.suptitle("Store usage và mức độ tập trung cửa hàng", fontsize=16)
    return figure


def plot_transaction_discount_usage(
    baskets: pl.DataFrame,
    households: pl.DataFrame,
) -> Figure:
    """Show discount incidence, basket-value contrast, and discount intensity."""
    require_columns(
        baskets,
        ("has_discount", "customer_spend", "discount_rate", "value_is_valid"),
    )
    require_columns(households, ("discounted_basket_rate",))
    valid = baskets.filter(pl.col("value_is_valid"))
    spend_cap = valid["customer_spend"].quantile(0.99)
    discounted = valid.filter(pl.col("has_discount"))["customer_spend"].to_numpy()
    regular = valid.filter(~pl.col("has_discount"))["customer_spend"].to_numpy()
    rates = numeric_values(households, "discounted_basket_rate")
    relation = valid.filter(
        pl.col("has_discount")
        & pl.col("discount_rate").is_not_null()
        & (pl.col("customer_spend") <= spend_cap)
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    axes[0].hist(rates, bins=np.linspace(0, 1, 26), color=GREEN)
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set(
        title="Discount incidence theo household",
        xlabel="Tỷ lệ basket có discount",
        ylabel="Số household",
    )
    axes[1].boxplot(
        [regular[regular <= spend_cap], discounted[discounted <= spend_cap]],
        tick_labels=["Không discount", "Có discount"],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#D1FAE5"},
        medianprops={"color": ORANGE, "linewidth": 2},
    )
    axes[1].set(title="Basket value theo discount usage", ylabel="Customer spend")
    axes[2].hexbin(
        relation["discount_rate"].to_numpy(),
        relation["customer_spend"].to_numpy(),
        gridsize=30,
        mincnt=1,
        cmap="Purples",
        bins="log",
    )
    axes[2].xaxis.set_major_formatter(PercentFormatter(1))
    axes[2].set(
        title="Discount intensity và basket value",
        xlabel="Discount rate",
        ylabel="Customer spend (đến p99)",
    )
    for axis in axes:
        finish_axis(axis)
    figure.suptitle(
        "Transaction-level discount usage — mô tả, không suy luận causal", fontsize=16
    )
    return figure


def plot_cross_behavior(households: pl.DataFrame) -> Figure:
    """Draw a Spearman heatmap across the core notebook-03 behaviors."""
    columns = (
        "basket_count",
        "avg_basket_value",
        "total_spend",
        "avg_products_per_basket",
        "unique_stores",
        "primary_store_share",
        "discounted_basket_rate",
    )
    require_columns(households, columns)
    ranked = households.select(
        pl.col(column).rank(method="average").alias(column) for column in columns
    ).drop_nulls()
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.corrcoef(
            np.column_stack([ranked[column].to_numpy() for column in columns]),
            rowvar=False,
        )
    correlation = np.nan_to_num(correlation, nan=0.0)
    np.fill_diagonal(correlation, 1.0)
    labels = (
        "Frequency",
        "Avg basket value",
        "Total spend",
        "Products / basket",
        "Unique stores",
        "Primary-store share",
        "Discounted-basket rate",
    )
    figure, axis = plt.subplots(figsize=(9, 7.5), constrained_layout=True)
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r")
    axis.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(
                column,
                row,
                f"{correlation[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(correlation[row, column]) > 0.55 else "#111827",
                fontsize=9,
            )
    figure.colorbar(image, ax=axis, label="Spearman correlation", shrink=0.8)
    axis.set_title("Quan hệ giữa các đặc điểm hành vi", fontsize=16)
    return figure


def plot_behavioral_patterns(households: pl.DataFrame) -> Figure:
    """Create descriptive frequency-value quadrants and compare their spending."""
    require_columns(
        households,
        ("basket_count", "avg_basket_value", "total_spend"),
    )
    valid = households.filter(
        (pl.col("total_spend") > 0) & pl.col("avg_basket_value").is_not_null()
    )
    frequency_cut = valid["basket_count"].median()
    value_cut = valid["avg_basket_value"].median()
    patterned = valid.with_columns(
        pl.when(pl.col("basket_count") >= frequency_cut)
        .then(
            pl.when(pl.col("avg_basket_value") >= value_cut)
            .then(pl.lit("Frequent high-value"))
            .otherwise(pl.lit("Frequent small-basket"))
        )
        .otherwise(
            pl.when(pl.col("avg_basket_value") >= value_cut)
            .then(pl.lit("Infrequent stock-up"))
            .otherwise(pl.lit("Low engagement"))
        )
        .alias("pattern")
    )
    labels = (
        "Low engagement",
        "Infrequent stock-up",
        "Frequent small-basket",
        "Frequent high-value",
    )
    colors = (GRAY, ORANGE, BLUE, GREEN)
    figure, (scatter_axis, box_axis) = plt.subplots(
        1, 2, figsize=(13, 5.5), constrained_layout=True
    )
    for label, color in zip(labels, colors, strict=True):
        group = patterned.filter(pl.col("pattern") == label)
        scatter_axis.scatter(
            group["basket_count"].to_numpy(),
            group["avg_basket_value"].to_numpy(),
            s=14,
            alpha=0.45,
            color=color,
            label=label,
            edgecolors="none",
        )
    scatter_axis.axvline(frequency_cut, color="#111827", linestyle="--")
    scatter_axis.axhline(value_cut, color="#111827", linestyle="--")
    scatter_axis.set_xscale("log")
    scatter_axis.set_yscale("log")
    scatter_axis.set(
        title="Bốn pattern frequency × basket value",
        xlabel="Basket count (log)",
        ylabel="Average basket value (log)",
    )
    scatter_axis.legend(frameon=False, fontsize=8)
    spend_groups = [
        patterned.filter(pl.col("pattern") == label)["total_spend"].to_numpy()
        for label in labels
    ]
    box_axis.boxplot(
        spend_groups,
        tick_labels=[
            "Low\nengagement",
            "Stock-up",
            "Frequent\nsmall",
            "Frequent\nhigh",
        ],
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#111827", "linewidth": 2},
    )
    box_axis.set_yscale("log")
    box_axis.set(
        title="Spending khác nhau giữa các pattern",
        ylabel="Total spend (log)",
    )
    finish_axis(scatter_axis)
    finish_axis(box_axis)
    figure.suptitle(
        "Behavioral patterns mô tả — chưa phải segmentation cuối cùng", fontsize=16
    )
    return figure

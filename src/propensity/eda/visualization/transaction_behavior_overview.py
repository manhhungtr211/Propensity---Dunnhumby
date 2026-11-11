"""Frequency, basket, and spending views for transaction behavior."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from propensity.eda.visualization.transaction_behavior_common import (
    GREEN,
    ORANGE,
    PURPLE,
    finish_axis,
    numeric_values,
    require_columns,
)


def plot_purchase_frequency(households: pl.DataFrame) -> Figure:
    """Show the distribution and cumulative coverage of basket frequency."""
    require_columns(households, ("basket_count",))
    values = numeric_values(households, "basket_count")
    edges = np.geomspace(1, max(2, values.max() + 1), 24)
    ordered = np.sort(values)
    cumulative = np.arange(1, ordered.size + 1) / ordered.size

    figure, (hist_axis, cumulative_axis) = plt.subplots(
        1, 2, figsize=(12, 5.2), constrained_layout=True
    )
    hist_axis.hist(values, bins=edges, color=GREEN, edgecolor="white")
    hist_axis.set_xscale("log")
    hist_axis.set(
        title="Phân phối purchase frequency",
        xlabel="Số basket / household (log scale)",
        ylabel="Số household",
    )
    cumulative_axis.plot(ordered, cumulative, color=ORANGE, linewidth=2.5)
    cumulative_axis.set_xscale("log")
    cumulative_axis.yaxis.set_major_formatter(PercentFormatter(1))
    cumulative_axis.set(
        title="Tỷ lệ household tích lũy",
        xlabel="Số basket / household (log scale)",
        ylabel="Tỷ lệ household",
        ylim=(0, 1.02),
    )
    for axis in (hist_axis, cumulative_axis):
        finish_axis(axis)
    figure.suptitle("Purchase frequency trong toàn bộ observation window", fontsize=16)
    return figure


def plot_basket_size_value_structure(baskets: pl.DataFrame) -> Figure:
    """Draw basket product count, department breadth, value, and their relation."""
    require_columns(
        baskets,
        ("product_count", "category_count", "customer_spend", "value_is_valid"),
    )
    valid = baskets.filter(pl.col("value_is_valid"))
    products = numeric_values(baskets, "product_count")
    categories = numeric_values(baskets, "category_count")
    spend = numeric_values(valid, "customer_spend")
    product_cap = np.quantile(products, 0.99)
    category_cap = np.quantile(categories, 0.99)
    spend_cap = np.quantile(spend, 0.99)
    relation = valid.filter(
        (pl.col("product_count") <= product_cap)
        & (pl.col("customer_spend") <= spend_cap)
    )

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes[0, 0].hist(products[products <= product_cap], bins=35, color=GREEN)
    axes[0, 0].set(
        title="Basket size", xlabel="Số product distinct", ylabel="Số basket"
    )
    axes[0, 1].hist(categories[categories <= category_cap], bins=25, color=PURPLE)
    axes[0, 1].set(
        title="Basket breadth",
        xlabel="Số department distinct",
        ylabel="Số basket",
    )
    axes[1, 0].hist(spend[spend <= spend_cap], bins=40, color=ORANGE)
    axes[1, 0].set(
        title="Basket value hợp lệ",
        xlabel="Customer spend",
        ylabel="Số basket",
    )
    axes[1, 1].hexbin(
        relation["product_count"].to_numpy(),
        relation["customer_spend"].to_numpy(),
        gridsize=35,
        mincnt=1,
        cmap="YlGn",
        bins="log",
    )
    axes[1, 1].set(
        title="Size và value của basket",
        xlabel="Số product distinct",
        ylabel="Customer spend",
    )
    for axis in axes.flat:
        finish_axis(axis)
    figure.suptitle(
        "Basket size, value và cấu trúc cơ bản (hiển thị đến p99)", fontsize=16
    )
    return figure


def plot_household_spending(households: pl.DataFrame) -> Figure:
    """Visualize household spending, concentration, and its decomposition."""
    require_columns(
        households,
        ("total_spend", "basket_count", "avg_basket_value"),
    )
    spend = numeric_values(households, "total_spend")
    spend = spend[spend > 0]
    spend_cap = np.quantile(spend, 0.99)
    descending = np.sort(spend)[::-1]
    household_share = np.arange(1, descending.size + 1) / descending.size
    spend_share = np.cumsum(descending) / descending.sum()
    relation = households.filter(
        (pl.col("total_spend") > 0) & pl.col("avg_basket_value").is_not_null()
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    axes[0].hist(spend[spend <= spend_cap], bins=40, color=GREEN)
    axes[0].set(
        title="Phân phối household spending",
        xlabel="Total spend (đến p99)",
        ylabel="Số household",
    )
    axes[1].plot(household_share, spend_share, color=ORANGE, linewidth=2.5)
    axes[1].plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--")
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set(
        title="Mức độ tập trung chi tiêu",
        xlabel="Top household tích lũy",
        ylabel="Chi tiêu tích lũy",
    )
    axes[2].hexbin(
        relation["basket_count"].to_numpy(),
        relation["avg_basket_value"].to_numpy(),
        gridsize=30,
        mincnt=1,
        cmap="Blues",
        bins="log",
        xscale="log",
        yscale="log",
    )
    axes[2].set(
        title="Hai thành phần tạo total spend",
        xlabel="Basket count (log)",
        ylabel="Average basket value (log)",
    )
    for axis in axes:
        finish_axis(axis)
    figure.suptitle(
        "Household spending = frequency × average basket value", fontsize=16
    )
    return figure

"""Frequency, basket, and spending views for transaction behavior."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from propensity.eda.visualization.transaction_behavior_common import (
    BASE,
    BASE_DARK,
    DENSITY_CMAP,
    HIGHLIGHT,
    REFERENCE,
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
    hist_axis.hist(values, bins=edges, color=BASE, edgecolor="white")
    mean_frequency = float(values.mean())
    hist_axis.axvline(
        mean_frequency,
        color=HIGHLIGHT,
        linewidth=2,
        linestyle="--",
        label=f"Mean = {mean_frequency:,.1f}",
    )
    hist_axis.set_xscale("log")
    hist_axis.set(
        title="Phân phối purchase frequency",
        xlabel="Số basket / household (log scale)",
        ylabel="Số household",
    )
    hist_axis.legend(frameon=False)
    cumulative_axis.plot(ordered, cumulative, color=BASE_DARK, linewidth=2.5)
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


def plot_basket_size_value_structure(
    baskets: pl.DataFrame,
    category_penetration: pl.DataFrame,
) -> Figure:
    """Draw basket size, breadth, department penetration, and value."""
    require_columns(
        baskets,
        ("product_count", "category_count", "customer_spend", "value_is_valid"),
    )
    require_columns(category_penetration, ("category", "basket_share"))
    products = numeric_values(baskets, "product_count")
    categories = numeric_values(baskets, "category_count")
    spend = numeric_values(
        baskets.filter(pl.col("value_is_valid")),
        "customer_spend",
    )
    product_cap = np.quantile(products, 0.99)
    category_cap = np.quantile(categories, 0.99)
    product_mean = float(products.mean())
    spend_mean = float(spend.mean())
    spend_p97 = float(np.quantile(spend, 0.97))
    penetration = category_penetration.sort("basket_share")

    figure, axes = plt.subplots(2, 2, figsize=(16, 10.5), constrained_layout=True)
    axes[0, 0].hist(products[products <= product_cap], bins=35, color=BASE)
    axes[0, 0].axvline(
        product_mean,
        color=HIGHLIGHT,
        linewidth=2,
        linestyle="--",
        label=f"Mean = {product_mean:,.1f}",
    )
    axes[0, 0].set(
        title="Basket size",
        xlabel="Số product distinct",
        ylabel="Số basket",
    )
    axes[0, 0].legend(frameon=False)
    axes[0, 1].hist(categories[categories <= category_cap], bins=25, color=BASE)
    axes[0, 1].set(
        title="Basket breadth",
        xlabel="Số department distinct",
        ylabel="Số basket",
    )
    department_labels = penetration["category"].to_list()
    department_colors = [
        BASE_DARK if category == "Other" else BASE for category in department_labels
    ]
    axes[1, 0].barh(
        department_labels,
        penetration["basket_share"].to_list(),
        color=department_colors,
    )
    axes[1, 0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 0].set(
        title="Basket penetration by department",
        xlabel="Share of all baskets",
        ylabel="Department",
    )
    central_spend = spend[spend <= spend_p97]
    counts, edges = np.histogram(central_spend, bins=35)
    widths = np.diff(edges)
    axes[1, 1].bar(
        edges[:-1],
        counts,
        width=widths,
        align="edge",
        color=BASE,
        edgecolor="white",
    )
    overflow_width = widths[-1]
    overflow_left = edges[-1] + overflow_width * 0.5
    overflow_center = overflow_left + overflow_width / 2
    axes[1, 1].bar(
        overflow_left,
        int((spend > spend_p97).sum()),
        width=overflow_width,
        align="edge",
        color=BASE_DARK,
        edgecolor="white",
    )
    axes[1, 1].axvline(
        spend_mean,
        color=HIGHLIGHT,
        linewidth=2,
        linestyle="--",
        label=f"Mean = {spend_mean:,.1f}",
    )
    numeric_ticks = [tick for tick in axes[1, 1].get_xticks() if tick <= edges[-1]]
    axes[1, 1].set_xticks(
        [*numeric_ticks, overflow_center],
        [*[f"{tick:,.0f}" for tick in numeric_ticks], "> p95"],
    )
    axes[1, 1].set_xlim(edges[0], overflow_left + overflow_width)
    axes[1, 1].set(
        title="Phân phối giá trị trên mỗi basket",
        xlabel="Customer spend / basket",
        ylabel="Số basket",
    )
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        finish_axis(axis)
    figure.suptitle("Basket size, breadth, category penetration và value", fontsize=18)
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
    axes[0].hist(spend[spend <= spend_cap], bins=40, color=BASE)
    axes[0].set(
        title="Phân phối household spending",
        xlabel="Total spend (đến p99)",
        ylabel="Số household",
    )
    axes[1].plot(household_share, spend_share, color=BASE_DARK, linewidth=2.5)
    axes[1].plot([0, 1], [0, 1], color=REFERENCE, linestyle="--")
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
        cmap=DENSITY_CMAP,
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
        "Household spending = frequency × average basket value",
        fontsize=16,
    )
    return figure

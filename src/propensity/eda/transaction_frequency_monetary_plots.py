"""Box plots for transaction frequency and monetary features by label."""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from propensity.eda.transaction_frequency_monetary_label import (
    FEATURE_COLUMNS,
    FEATURE_TITLES,
)

LABEL_PALETTE: Final = {0: "#64748B", 1: "#F97316"}
MONETARY_FEATURES: Final = {
    "total_sales_value_90d",
    "avg_spend_per_basket_90d",
    "median_basket_value_90d",
    "max_basket_value_90d",
    "spend_per_active_day_90d",
}


def plot_frequency_monetary_boxplots(feature_panel: pd.DataFrame) -> Figure:
    """Plot one label comparison per feature on its natural scale."""
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(15, 13),
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
            whis=(5, 95),
            showfliers=False,
            ax=axis,
        )
        axis.set_title(FEATURE_TITLES[feature])
        axis.set_xlabel("Label (0 = Negative, 1 = Positive)")
        axis.set_ylabel(
            "SALES_VALUE" if feature in MONETARY_FEATURES else "Basket intensity"
        )
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)

    for axis in axes.flat[len(FEATURE_COLUMNS) :]:
        axis.set_visible(False)
    figure.suptitle(
        "Transaction frequency và monetary features theo label",
        fontsize=15,
    )
    return figure

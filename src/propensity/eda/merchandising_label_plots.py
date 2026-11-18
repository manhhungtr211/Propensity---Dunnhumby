"""Positive-rate plots for display and mailer exposure."""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from propensity.eda.merchandising_label import EXPOSURE_FEATURES, FEATURE_TITLES

EXPOSURE_PALETTE: Final = {False: "#64748B", True: "#F97316"}


def plot_merchandising_positive_rates(feature_panel: pd.DataFrame) -> Figure:
    """Compare Positive rates for unexposed and exposed household-cutoffs."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for axis, feature in zip(axes, EXPOSURE_FEATURES, strict=True):
        plot_data = feature_panel[["label", feature]].dropna().copy()
        plot_data[feature] = plot_data[feature].astype("bool")
        summary = (
            plot_data.groupby(feature, as_index=False)
            .agg(positive_rate=("label", "mean"), n_observations=("label", "size"))
            .sort_values(feature)
        )
        labels = summary[feature].map({False: "Không exposure", True: "Có exposure"})
        bars = axis.barh(
            labels,
            summary["positive_rate"],
            color=[EXPOSURE_PALETTE[value] for value in summary[feature]],
        )
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
        axis.set_title(FEATURE_TITLES[feature])
        axis.set_xlabel("Tỷ lệ label Positive")
        axis.set_ylabel("")
        axis.set_xlim(0, 1)
        axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        axis.grid(axis="y", visible=False)
        axis.spines[["top", "right", "left"]].set_visible(False)
    figure.suptitle("Merchandising exposure và propensity", fontsize=15)
    return figure

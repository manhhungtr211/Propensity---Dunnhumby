"""Stacked label-mix plots for historical departments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from propensity.eda.department_grouping import build_department_group_shares

LABEL_PALETTE: Final = {"Negative": "#64748B", "Positive": "#F97316"}


@dataclass(frozen=True, slots=True)
class LabelMixPlotSpec:
    """Text and sizing for a stacked label-mix chart."""

    title: str
    x_label: str
    row_height: float


def plot_department_label_mix_stacked(summary: pd.DataFrame) -> Figure:
    """Plot label percentages for each department, ordered by Positive share."""
    department_data = summary.loc[summary["dimension"] == "department"]
    counts = department_data.pivot_table(
        index="category",
        columns="label",
        values="n_observations",
        aggfunc="sum",
        fill_value=0,
    ).reindex(columns=[0, 1], fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0).rename(
        columns={0: "Negative", 1: "Positive"}
    )
    shares = shares.sort_values("Positive", ascending=True)
    return _plot_label_mix(
        shares,
        LabelMixPlotSpec(
            title="Cơ cấu Negative/Positive trong từng historical department",
            x_label="Tỷ lệ label trong department",
            row_height=0.38,
        ),
    )


def plot_department_group_label_mix_stacked(summary: pd.DataFrame) -> Figure:
    """Plot mean label percentages for the requested department groups."""
    return _plot_label_mix(
        build_department_group_shares(summary),
        LabelMixPlotSpec(
            title="Cơ cấu Negative/Positive trung bình theo nhóm department",
            x_label="Tỷ lệ label trung bình của các department trong nhóm",
            row_height=0.75,
        ),
    )


def _plot_label_mix(
    shares: pd.DataFrame,
    spec: LabelMixPlotSpec,
) -> Figure:
    figure, axis = plt.subplots(
        figsize=(13, max(6, spec.row_height * len(shares))),
        layout="constrained",
    )
    negative_bars = axis.barh(
        shares.index,
        shares["Negative"],
        color=LABEL_PALETTE["Negative"],
        label="Negative",
    )
    positive_bars = axis.barh(
        shares.index,
        shares["Positive"],
        left=shares["Negative"],
        color=LABEL_PALETTE["Positive"],
        label="Positive",
    )
    for bars, values in (
        (negative_bars, shares["Negative"]),
        (positive_bars, shares["Positive"]),
    ):
        axis.bar_label(
            bars,
            labels=[f"{value:.1%}" if value >= 0.05 else "" for value in values],
            label_type="center",
            color="white",
            fontsize=8,
        )
    axis.set_title(spec.title)
    axis.set_xlabel(spec.x_label)
    axis.set_ylabel("")
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.legend(title="Label", frameon=False, ncol=2, loc="lower right")
    axis.grid(axis="y", visible=False)
    axis.spines[["top", "right", "left"]].set_visible(False)
    return figure

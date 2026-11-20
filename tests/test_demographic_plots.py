"""Unit tests for demographic plotting functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from propensity.eda.demographic_plots import (
    plot_attribute_distribution,
    plot_attribute_excess,
    plot_demographic_coverage,
)


def test_plot_demographic_coverage() -> None:
    panel = pd.DataFrame(
        {
            "has_demographic": [1, 1, 0, 0, 1, 0],
            "F_bin": pd.Categorical(["F1", "F2", "F1", "F3", "F2", "F3"]),
        }
    )
    fig = plot_demographic_coverage(panel)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_attribute_distribution() -> None:
    demo_grouped = pd.DataFrame(
        {
            "household_key": list(range(1, 11)),
            "demo_age": pd.Categorical(
                ["19-34", "19-34", "35-44", "35-44", "45-54", "45-54", "45-54", "55+", "55+", "55+"],
                categories=["19-34", "35-44", "45-54", "55+"],
                ordered=True,
            ),
        }
    )
    fig = plot_attribute_distribution(demo_grouped, "age")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_attribute_excess() -> None:
    table = pd.DataFrame(
        {
            "attribute": ["Độ tuổi", "Độ tuổi", "Thu nhập", "Thu nhập"],
            "level": ["19-34", "35-44", "<35K", "35-49K"],
            "n_households": [188, 194, 212, 172],
            "excess_pp": [4.3, 4.7, 4.2, 4.5],
            "verdict": ["Insight", "Insight", "Insight", "Insight"],
        }
    )
    fig = plot_attribute_excess(table)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

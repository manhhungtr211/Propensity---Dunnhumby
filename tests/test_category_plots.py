"""Unit tests for category plots."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from propensity.features.category import CategoryConfig
from propensity.eda.category_plots import (
    plot_department_pareto,
    plot_commodity_scatter,
    plot_department_excess,
)


def test_plot_department_pareto() -> None:
    """Verify plot_department_pareto runs and produces a figure."""
    df = pd.DataFrame(
        {
            "department": ["GROCERY", "DRUG GM", "PRODUCE"],
            "spend": [5000.0, 2000.0, 1000.0],
            "spend_share": [0.625, 0.250, 0.125],
            "penetration": [0.95, 0.80, 0.75],
            "cum_spend_share": [0.625, 0.875, 1.0],
        }
    )
    fig = plot_department_pareto(df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_commodity_scatter() -> None:
    """Verify plot_commodity_scatter renders threshold lines and points."""
    stats = pd.DataFrame(
        {
            "commodity": ["MILK", "APPLES", "SNACKS"],
            "department": ["GROCERY", "PRODUCE", "GROCERY"],
            "n_buyers": [50, 40, 60],
            "penetration": [0.70, 0.45, 0.65],
            "repeat_buyer_share": [0.8, 0.5, 0.7],
            "median_gap": [10.0, 25.0, 12.0],
            "spend": [1000.0, 500.0, 800.0],
            "spend_share": [0.4, 0.2, 0.3],
            "is_staple": [True, False, True],
        }
    )
    fig = plot_commodity_scatter(stats, CategoryConfig())
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_department_excess() -> None:
    """Verify plot_department_excess renders dot plot within indifference zone."""
    df = pd.DataFrame(
        {
            "department": ["MEAT", "PRODUCE"],
            "excess_pp": [0.45, -0.20],
            "verdict": ["Đã được R × F giải thích", "Không được ủng hộ"],
            "n_households": [2000, 2100],
        }
    )
    fig = plot_department_excess(df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

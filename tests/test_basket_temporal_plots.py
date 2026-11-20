"""Unit tests for basket and temporal plotting functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from propensity.eda.basket_temporal_plots import (
    plot_basket_size_bands,
    plot_cutoff_return_rate,
    plot_products_per_basket_hist,
    plot_weekly_active,
)


def test_plot_basket_size_bands() -> None:
    bands = pd.DataFrame(
        {
            "band": ["1–2", "3–5", "6–10", "11–20", "21+"],
            "n_baskets": [100, 200, 300, 150, 50],
            "pct_baskets": [0.125, 0.25, 0.375, 0.1875, 0.0625],
            "total_spend": [1000.0, 3000.0, 6000.0, 4500.0, 2000.0],
            "pct_spend": [0.0606, 0.1818, 0.3636, 0.2727, 0.1212],
            "median_spend": [10.0, 15.0, 20.0, 30.0, 40.0],
        }
    )
    fig = plot_basket_size_bands(bands)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_products_per_basket_hist() -> None:
    baskets = pd.DataFrame(
        {
            "n_products": [1, 2, 2, 3, 4, 5, 6, 8, 12, 25, 30],
        }
    )
    fig = plot_products_per_basket_hist(baskets)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_weekly_active() -> None:
    weekly = pd.DataFrame(
        {
            "WEEK_NO": list(range(1, 25)),
            "active_hh": [500 + i * 10 for i in range(24)],
            "n_baskets": [1200 + i * 20 for i in range(24)],
            "min_day": [i * 7 for i in range(24)],
            "max_day": [(i + 1) * 7 - 1 for i in range(24)],
        }
    )
    fig = plot_weekly_active(weekly)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_cutoff_return_rate() -> None:
    rates = pd.DataFrame(
        {
            "cutoff_day": [140, 147, 154, 161, 168],
            "return_rate": [0.52, 0.54, 0.51, 0.53, 0.68],
            "robust_z": [0.1, 0.5, -0.2, 0.2, 4.1],
            "is_unusual": [False, False, False, False, True],
        }
    )
    fig = plot_cutoff_return_rate(rates)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

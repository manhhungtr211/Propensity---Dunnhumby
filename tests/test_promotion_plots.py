"""Unit tests for promotion plots."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from propensity.eda.promotion_plots import (
    plot_campaign_overview,
    plot_campaign_reach,
    plot_department_promo,
    plot_promo_dept_week,
    plot_weekly_promo,
)


def test_plot_weekly_promo() -> None:
    """Verify plot_weekly_promo creates a 2-panel figure without dual y-axes."""
    weekly = pd.DataFrame(
        {
            "WEEK_NO": [1, 2, 3],
            "sales": [1000.0, 1500.0, 1200.0],
            "promo_spend_share": [0.25, 0.35, 0.30],
            "discount_depth": [0.10, 0.12, 0.11],
        }
    )
    camp = pd.DataFrame(
        [
            {"type": "Type A", "start": 5, "end": 18},
        ]
    )
    fig = plot_weekly_promo(weekly, camp)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2  # exactly 2 stacked subplots sharing x
    plt.close(fig)


def test_plot_department_and_matrix_promo() -> None:
    """Verify department promo bar and matrix plots."""
    dept = pd.DataFrame(
        {
            "department": ["GROCERY", "PRODUCE"],
            "promo_spend_share": [0.45, 0.20],
            "spend_share": [0.60, 0.15],
        }
    )
    fig_d = plot_department_promo(dept)
    assert isinstance(fig_d, plt.Figure)
    plt.close(fig_d)

    matrix = pd.DataFrame(
        [[0.4, 0.5], [0.2, 0.3]],
        index=["GROCERY", "PRODUCE"],
        columns=[1, 2],
    )
    fig_m = plot_promo_dept_week(matrix)
    assert isinstance(fig_m, plt.Figure)
    plt.close(fig_m)


def test_plot_campaign_overview_and_reach() -> None:
    """Verify campaign overview and reach plots."""
    summary = pd.DataFrame(
        [
            {"CAMPAIGN": 1, "type": "Type A", "start": 10, "end": 40, "duration": 31, "n_hh_received": 500, "n_hh_redeemed": 80, "redemption_rate": 0.16},
            {"CAMPAIGN": 2, "type": "Type B", "start": 50, "end": 80, "duration": 31, "n_hh_received": 300, "n_hh_redeemed": 30, "redemption_rate": 0.10},
        ]
    )
    t_sum = pd.DataFrame(
        [
            {"type": "Type A", "redemption_rate": 0.16, "total_hh_received": 500, "total_hh_redeemed": 80},
            {"type": "Type B", "redemption_rate": 0.10, "total_hh_received": 300, "total_hh_redeemed": 30},
        ]
    )
    fig_c = plot_campaign_overview(summary, t_sum)
    assert isinstance(fig_c, plt.Figure)
    assert len(fig_c.axes) == 2
    plt.close(fig_c)

    reach = {
        "by_f": pd.DataFrame({"F_bin": ["Q1", "Q2"], "reach_rate": [0.2, 0.6]}),
        "by_r": pd.DataFrame({"recency_bin": ["1–7", "8–14"], "reach_rate": [0.5, 0.3]}),
    }
    fig_r = plot_campaign_reach(reach)
    assert isinstance(fig_r, plt.Figure)
    assert len(fig_r.axes) == 2
    plt.close(fig_r)

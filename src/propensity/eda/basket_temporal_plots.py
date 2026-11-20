"""Visualizations for basket size distributions, temporal dynamics, and cutoff stability."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from propensity.viz.style import (
    ACCENT,
    BAR,
    BOX_BG,
    FIGSIZE_FULL,
    FIGSIZE_HALF,
    MEDIAN,
    PERCENTILE,
    PRIMARY,
    REF,
    SERIES_2,
    TITLE,
    add_median_lines,
    add_ref_line,
    annotate_box,
    fmt_pct,
    plot_rate_and_excess,
    set_style,
)


def plot_basket_size_bands(bands: pd.DataFrame) -> plt.Figure:
    """Plot dual-bar chart comparing % of baskets vs % of spend across 5 product-count bands.

    Parameters
    ----------
    bands : pd.DataFrame
        Output of basket_size_bands containing band, pct_baskets, pct_spend, n_baskets, median_spend.

    Returns
    -------
    plt.Figure
    """
    set_style()
    df = bands.copy()

    fig, ax = plt.subplots(figsize=FIGSIZE_FULL, dpi=150)

    n_bands = len(df)
    x = np.arange(n_bands)
    bar_width = 0.35

    pct_baskets = df["pct_baskets"] * 100.0
    pct_spend = df["pct_spend"] * 100.0

    bars1 = ax.bar(
        x - bar_width / 2.0,
        pct_baskets,
        width=bar_width,
        color=BAR,
        edgecolor="white",
        linewidth=1.2,
        alpha=0.85,
        label="% Lượt mua (Baskets)",
        zorder=3,
    )

    bars2 = ax.bar(
        x + bar_width / 2.0,
        pct_spend,
        width=bar_width,
        color=PRIMARY,
        edgecolor="white",
        linewidth=1.2,
        alpha=0.85,
        label="% Doanh số (Spend)",
        zorder=3,
    )

    # Annotate percentages on top of bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.0,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=TITLE,
        )

    for bar in bars2:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.0,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=PRIMARY,
        )

    # Secondary text under band name with median spend
    xtick_labels = []
    for _, row in df.iterrows():
        b_name = str(row["band"])
        med_s = row["median_spend"]
        xtick_labels.append(f"{b_name} SP\n(Med: ${med_s:.1f})")

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=9.0)
    ax.set_xlabel("Quy mô giỏ hàng (Số sản phẩm khác nhau trong giỏ)", fontsize=9.5, fontweight="bold", labelpad=8)
    ax.set_ylabel("Tỷ trọng (%)", fontsize=9.5, fontweight="bold")

    max_val = max(pct_baskets.max(), pct_spend.max())
    ax.set_ylim(0, max(58.0, max_val * 1.25))
    fmt_pct(ax, axis="y")

    total_baskets = int(df["n_baskets"].sum())
    total_spend = df["total_spend"].sum() if "total_spend" in df.columns else 0.0
    annotate_box(
        ax,
        f"Tổng: {total_baskets:,} giỏ hàng hàng hóa\nDoanh số: ${total_spend:,.0f}",
        loc="upper left",
        coords=(0.03, 0.74),
    )

    # Dynamic headline
    small_baskets_pct = pct_baskets.iloc[:2].sum() if len(pct_baskets) >= 2 else 0.0
    small_spend_pct = pct_spend.iloc[:2].sum() if len(pct_spend) >= 2 else 0.0
    title = f"Q4a: Giỏ nhỏ 1–5 sản phẩm chiếm {small_baskets_pct:.1f}% lượt mua nhưng chỉ đóng góp {small_spend_pct:.1f}% doanh số"
    ax.set_title(title, fontsize=12, fontweight="bold", color=TITLE, pad=12)
    ax.legend(loc="upper left")

    fig.tight_layout()
    return fig


def plot_products_per_basket_hist(baskets: pd.DataFrame) -> plt.Figure:
    """Plot histogram of distinct merchandise products per basket clipped at p99.5.

    Parameters
    ----------
    baskets : pd.DataFrame
        Output of basket_table containing n_products.

    Returns
    -------
    plt.Figure
    """
    set_style()
    df = baskets[baskets["n_products"] >= 1].copy()

    fig, ax = plt.subplots(figsize=FIGSIZE_HALF, dpi=150)

    s = df["n_products"].dropna()
    total_baskets = len(s)
    p995 = float(np.percentile(s, 99.5))
    s_clipped = s[s <= p995]
    n_clipped = total_baskets - len(s_clipped)

    max_bins = int(min(p995, 45))
    bins = np.linspace(1, p995, max_bins)

    ax.hist(
        s_clipped,
        bins=bins,
        color=BAR,
        edgecolor="white",
        linewidth=1.0,
        alpha=0.85,
        zorder=3,
    )

    add_median_lines(ax, s, fmt=".1f")

    annotate_box(
        ax,
        f"Tổng: {total_baskets:,} giỏ\nCắt đuôi >p99.5 ({p995:.0f} SP): {n_clipped:,} giỏ",
        loc="upper right",
    )

    ax.set_title("Q4a (Phụ): Phân bổ số sản phẩm trong giỏ hàng", fontsize=11, fontweight="bold", color=TITLE, pad=10)
    ax.set_xlabel("Số sản phẩm hàng hóa (n_products)", fontsize=9.0, fontweight="bold")
    ax.set_ylabel("Số lượng giỏ hàng", fontsize=9.0, fontweight="bold")
    ax.legend(loc="upper left")

    fig.tight_layout()
    return fig


def plot_weekly_active(weekly: pd.DataFrame) -> plt.Figure:
    """Plot active households over time with reference line at day 140 (first cutoff).

    Parameters
    ----------
    weekly : pd.DataFrame
        Output of weekly_activity containing WEEK_NO, active_hh, and min_day/max_day.

    Returns
    -------
    plt.Figure
    """
    set_style()
    df = weekly.copy()

    fig, ax = plt.subplots(figsize=FIGSIZE_FULL, dpi=150)

    # Map x to calendar DAY
    if "min_day" in df.columns:
        x = df["min_day"]
        xlabel = "Ngày trong năm (DAY)"
        first_cutoff_x = 140
    else:
        x = df["WEEK_NO"] * 7
        xlabel = "Ngày quy đổi từ tuần (WEEK_NO × 7)"
        first_cutoff_x = 140

    y = df["active_hh"]

    ax.plot(
        x,
        y,
        color=PRIMARY,
        linewidth=2.0,
        label="Hộ gia đình hoạt động (Active HH)",
        zorder=4,
    )

    # Reference vertical line at Day 140
    add_ref_line(ax, first_cutoff_x, label="Cutoff đầu tiên (Day 140)", is_vertical=True)

    # Mean active households after Day 140
    post_140 = df[x >= 140]
    if not post_140.empty:
        mean_post = float(post_140["active_hh"].mean())
        ax.axhline(
            mean_post,
            color=MEDIAN,
            linestyle="--",
            linewidth=1.4,
            alpha=0.85,
            label=f"Trung bình sau Day 140 ({mean_post:,.0f} hộ)",
            zorder=3,
        )

    ax.set_xlim(0, 720)
    ax.set_xlabel(xlabel, fontsize=9.5, fontweight="bold", labelpad=8)
    ax.set_ylabel("Số hộ có giao dịch (Active HH)", fontsize=9.5, fontweight="bold")
    ax.set_title("Q5a: Quy mô hộ gia đình hoạt động theo thời gian (Weekly Active Households)", fontsize=12, fontweight="bold", color=TITLE, pad=12)

    annotate_box(
        ax,
        "Dữ liệu ổn định và đạt quy mô sau Day 140\nTránh hiện tượng left-censoring giai đoạn đầu",
        loc="lower left",
    )

    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_cutoff_return_rate(rates: pd.DataFrame) -> plt.Figure:
    """Plot return rate across 81 cutoffs with p10-p90 shaded band and highlight unusual cutoffs.

    Parameters
    ----------
    rates : pd.DataFrame
        Output of cutoff_return_rates containing cutoff_day, return_rate, robust_z, is_unusual.

    Returns
    -------
    plt.Figure
    """
    set_style()
    df = rates.copy()

    fig, ax = plt.subplots(figsize=FIGSIZE_FULL, dpi=150)

    x = df["cutoff_day"]
    y_pct = df["return_rate"] * 100.0

    med_val = float(np.median(y_pct))
    p10_val, p90_val = float(np.percentile(y_pct, 10)), float(np.percentile(y_pct, 90))

    # Line of return rate
    ax.plot(
        x,
        y_pct,
        color=PRIMARY,
        linewidth=2.0,
        label="Tỷ lệ quay lại 7 ngày thực tế",
        zorder=4,
    )

    # Shaded p10-p90 band (strictly within observed cutoff range 140..700)
    ax.fill_between(
        x,
        p10_val,
        p90_val,
        color=BAR,
        alpha=0.30,
        label=f"Dải p10–p90 ({p10_val:.1f}% – {p90_val:.1f}%)",
        zorder=2,
    )

    # Median line within observed cutoff range
    ax.plot(
        [float(x.min()), float(x.max())],
        [med_val, med_val],
        color=MEDIAN,
        linestyle="--",
        linewidth=1.4,
        label=f"Median ({med_val:.1f}%)",
        zorder=3,
    )

    # Mark unusual cutoffs
    unusual = df[df["is_unusual"]]
    if not unusual.empty:
        ax.scatter(
            unusual["cutoff_day"],
            unusual["return_rate"] * 100.0,
            color=ACCENT,
            s=70,
            edgecolors="white",
            linewidths=1.2,
            label=f"Bất thường (|z| > 3.5, n={len(unusual)})",
            zorder=6,
        )
        for _, row in unusual.iterrows():
            cd = int(row["cutoff_day"])
            cr = float(row["return_rate"] * 100.0)
            ax.annotate(
                f"Day {cd}",
                xy=(cd, cr),
                xytext=(0, 10),
                textcoords="offset points",
                fontsize=8.0,
                fontweight="bold",
                color=ACCENT,
                ha="center",
            )

    ax.set_xlim(0, 720)  # Synchronized with plot_weekly_active
    ax.set_ylim(20.0, 70.0)
    fmt_pct(ax, axis="y")
    ax.set_xlabel("Ngày cutoff (DAY)", fontsize=9.5, fontweight="bold", labelpad=8)
    ax.set_ylabel("Tỷ lệ quay lại 7 ngày (%)", fontsize=9.5, fontweight="bold")
    ax.set_title("Q5a: Label ổn định: return rate 51,5–54,9% qua 81 cutoff", fontsize=12, fontweight="bold", color=TITLE, pad=12)

    n_unusual = len(unusual)
    summary_txt = (
        f"Tổng: {len(df)} cutoffs (Day 140..700)\n"
        f"Số cutoff bất thường (|z|>3.5): {n_unusual}"
    )
    annotate_box(ax, summary_txt, loc="lower left")

    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig

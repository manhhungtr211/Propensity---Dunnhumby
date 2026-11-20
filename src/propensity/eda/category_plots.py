"""Visualization functions for merchandise departments, staple commodities, and department excess effects."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from propensity.features.category import CategoryConfig
from propensity.viz.style import (
    BG_COLOR,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    GRAY_LIGHT,
    GRAY_MUTED,
    GRAY_NEUTRAL,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    set_style,
)


def plot_department_pareto(dept_summary: pd.DataFrame) -> plt.Figure:
    """Plot Pareto chart using 2 stacked subplots sharing x-axis to strictly eliminate dual y-axes.

    Upper panel: Department spend share (%) with penetration annotations.
    Lower panel: Cumulative spend share (%) with 80% Pareto threshold line.
    """
    set_style()
    df = dept_summary.sort_values("spend_share", ascending=False).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7.5), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]}
    )

    x = np.arange(len(df))
    bar_width = 0.65

    # Upper panel: Bar chart for spend share
    bars = ax1.bar(
        x,
        df["spend_share"] * 100.0,
        width=bar_width,
        color=COLOR_PRIMARY,
        alpha=0.90,
        label="Tỷ trọng doanh số ngành hàng (%)",
        zorder=3,
    )

    # Annotate penetration and spend share on top of bars
    for bar, (_, row) in zip(bars, df.iterrows()):
        h = bar.get_height()
        pen_early = row.get("penetration_early", row.get("penetration", 0.0)) * 100.0
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.2,
            f"{h:.1f}%\n(Phủ: {pen_early:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=TEXT_PRIMARY,
        )

    ax1.set_ylabel("Tỷ trọng doanh số (%)", fontsize=10.5, fontweight="bold")
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    ax1.set_ylim(0, max(65.0, float(df["spend_share"].max() * 100.0) + 14.0))
    ax1.set_title(
        "Cấu Trúc Doanh Số Ngành Hàng & Phân Bổ Pareto (Q2a: Department Pareto Overview)\n"
        "[Ghi chú trên cột: % Doanh số và Độ thâm nhập hộ (Phủ %) trên DAY ≤ 139]",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax1.legend(frameon=True, facecolor=BG_COLOR, edgecolor=GRAY_LIGHT, loc="upper right", fontsize=9)

    # Lower panel: Cumulative spend share
    cum_pct = df["cum_spend_share"] * 100.0
    ax2.plot(x, cum_pct, color=COLOR_SECONDARY, marker="o", linewidth=2.0, markersize=5, label="Doanh số tích lũy (%)", zorder=3)
    ax2.axhline(80.0, color="#ef4444", linestyle="-", linewidth=1.2, alpha=0.8, label="Ngưỡng Pareto 80%", zorder=2)

    ax2.set_ylabel("Tích lũy (%)", fontsize=10, fontweight="bold")
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    ax2.set_ylim(40.0, 105.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["department"], rotation=40, ha="right", fontsize=9.5, fontweight="semibold")
    ax2.legend(frameon=True, facecolor=BG_COLOR, edgecolor=GRAY_LIGHT, loc="lower right", fontsize=8.5)

    return fig


def plot_commodity_scatter(
    stats: pd.DataFrame,
    cfg: CategoryConfig | None = None,
    n_labels: int = 15,
) -> plt.Figure:
    """Plot commodity scatter: Penetration (x) vs Median purchase gap (y, log scale) to identify staple goods."""
    if cfg is None:
        cfg = CategoryConfig()

    df = stats.dropna(subset=["median_gap", "penetration"]).copy()

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

    # Marker sizes scaled by spend_share
    sizes = np.clip(df["spend_share"] * 4000.0 + 30.0, 30.0, 450.0)

    # Colors by is_staple
    staple_mask = df["is_staple"].astype(bool)
    ax.scatter(
        df.loc[~staple_mask, "penetration"] * 100.0,
        df.loc[~staple_mask, "median_gap"],
        s=sizes[~staple_mask],
        color="#94A3B8",
        alpha=0.6,
        edgecolors="#64748B",
        label="Không thiết yếu (Non-staple)",
        zorder=3,
    )
    ax.scatter(
        df.loc[staple_mask, "penetration"] * 100.0,
        df.loc[staple_mask, "median_gap"],
        s=sizes[staple_mask],
        color="#2563EB",
        alpha=0.88,
        edgecolors="#1D4ED8",
        label=f"Mặt hàng thiết yếu (Staple, n={int(staple_mask.sum())})",
        zorder=4,
    )

    # Threshold lines
    pen_thresh = cfg.min_penetration * 100.0
    gap_thresh = cfg.max_median_gap
    ax.axvline(pen_thresh, color="#EF4444", linestyle="--", linewidth=1.4, label=f"Ngưỡng Phủ ≥ {pen_thresh:.0f}%", zorder=2)
    ax.axhline(gap_thresh, color="#D97706", linestyle="--", linewidth=1.4, label=f"Ngưỡng Nhịp ≤ {gap_thresh:.0f} ngày", zorder=2)

    # Log scale for median gap
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    # Label top commodities by spend_share
    top_comm = df.nlargest(n_labels, "spend_share")
    for _, row in top_comm.iterrows():
        x_val = row["penetration"] * 100.0
        y_val = row["median_gap"]
        comm_name = row["commodity"]
        is_st = row["is_staple"]
        color = "#1E3A8A" if is_st else "#334155"
        weight = "bold" if is_st else "normal"
        ax.annotate(
            comm_name,
            xy=(x_val, y_val),
            xytext=(5, 2),
            textcoords="offset points",
            fontsize=7.5,
            color=color,
            fontweight=weight,
            zorder=5,
        )

    ax.set_xlabel("Độ phủ khách hàng trong cửa sổ chuẩn hóa DAY ≤ 139 (Penetration %)", fontsize=10.5, fontweight="bold", labelpad=8)
    ax.set_ylabel("Khoảng cách mua lặp lại trung vị (Median Gap in Days, Log scale)", fontsize=10.5, fontweight="bold", labelpad=8)
    ax.set_title(
        f"Xác Định Mặt Hàng Thiết Yếu (Commodity Staples) Trên Cửa Sổ DAY ≤ {cfg.staple_max_day}\n"
        f"[Tiêu chí: Mua ≥ {cfg.min_buyers} hộ, Phủ ≥ {pen_thresh:.0f}%, Nhịp mua lặp ≤ {gap_thresh:.0f} ngày]",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, linestyle=":", alpha=0.5, zorder=1)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E5E7EB", loc="upper right", fontsize=9)

    return fig


def plot_department_excess(table: pd.DataFrame) -> plt.Figure:
    """Plot dot plot of department excess effects over R x F expectation with +/- 3 pp indifference zone."""
    df = table.sort_values("excess_pp", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(5.5, 0.45 * len(df))), constrained_layout=True)

    y_pos = np.arange(len(df))

    # Indifference zone +/- 3 pp
    ax.axvspan(-3.0, 3.0, color="#F1F5F9", alpha=0.9, label="Vùng không có hiệu ứng ngoài R × F (±3 pp)", zorder=1)
    ax.axvline(0.0, color="#64748B", linestyle="--", linewidth=1.2, label="Mốc kỳ vọng R × F (0.0 pp)", zorder=2)

    # Dot plot with horizontal guide lines
    for y, (_, row) in zip(y_pos, df.iterrows()):
        val = row["excess_pp"]
        ax.hlines(y, xmin=min(0.0, val), xmax=max(0.0, val), color="#CBD5E1", linewidth=1.5, zorder=2)

    # Color dots by verdict
    colors = [
        "#10B981" if "Insight" in str(v) else ("#3B82F6" if "R × F" in str(v) else "#94A3B8")
        for v in df["verdict"]
    ]
    ax.scatter(df["excess_pp"], y_pos, color=colors, s=80, edgecolors="#1E293B", linewidths=1.2, zorder=4)

    # Annotate values and verdicts
    for y, (_, row) in zip(y_pos, df.iterrows()):
        val = row["excess_pp"]
        verdict = row.get("verdict", "")
        n_hh = row.get("n_households", 0)
        sign = "+" if val > 0 else ""
        text = f"{sign}{val:.2f} pp [{verdict}] (n={int(n_hh):,} hộ)"
        offset = 0.25 if val >= 0 else -0.25
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, y, text, va="center", ha=ha, fontsize=8, fontweight="bold", color="#1E293B")

    ax.set_yticks(y_pos)
    dept_names = df["department"] if "department" in df.columns else df.iloc[:, 0]
    ax.set_yticklabels(dept_names, fontsize=9.5, fontweight="semibold")
    ax.set_xlabel("Độ chênh tỷ lệ quay lại so với kỳ vọng R × F (Excess pp)", fontsize=10.5, fontweight="bold", labelpad=8)
    ax.set_title(
        "Hiệu Ứng Của Việc Từng Mua Các Ngành Hàng (Q3e: Department Flags vs R × F)\n"
        "[Kiểm định biến nhị phân đã mua: bought_<dept>_c == 1]",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=1)
    ax.set_xlim(min(-8.0, float(df["excess_pp"].min()) - 3.5), max(8.0, float(df["excess_pp"].max()) + 3.5))
    ax.legend(frameon=True, facecolor="white", edgecolor="#E5E7EB", loc="lower right", fontsize=8.5)

    return fig

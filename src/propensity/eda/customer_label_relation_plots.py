"""Visualization functions for customer variable-to-label relationships."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_f_strata(f_rates: pd.DataFrame, strata_matrix: pd.DataFrame) -> plt.Figure:
    """Plot return rate by F_bin and 2D strata heatmap with recency_bin (Q4)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # Panel 1: Bar chart (without error bars)
    bars = ax1.bar(
        range(len(f_rates)),
        f_rates["return_rate"] * 100.0,
        color="#2b5c8f",
        edgecolor="#1a365d",
        alpha=0.85,
        width=0.55,
    )
    ax1.set_xticks(range(len(f_rates)))
    ax1.set_xticklabels(f_rates["F_bin"], rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("Tỷ lệ quay lại trong 7 ngày (%)", fontsize=10, fontweight="bold")
    ax1.set_title("A. Tỷ lệ quay lại theo Tần suất F (lần/tuần)", fontsize=11, fontweight="bold", pad=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.set_ylim(0, 100)

    for bar, (_, row) in zip(bars, f_rates.iterrows()):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 2.0,
            f"{h:.1f}%\n(n={int(row['n']):,})",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#1a202c",
            fontweight="bold",
        )

    # Panel 2: Heatmap of F_bin x recency_bin
    # Sort strata rows to match f_rates order
    f_order = [str(x) for x in f_rates["F_bin"]]
    matrix_clean = strata_matrix.reindex(index=f_order).fillna(0.0) * 100.0

    im = ax2.imshow(matrix_clean.values, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    ax2.set_xticks(range(len(matrix_clean.columns)))
    ax2.set_xticklabels(matrix_clean.columns, fontsize=9)
    ax2.set_xlabel("Tầng Recency (ngày kể từ lần mua gần nhất)", fontsize=10, fontweight="bold")

    ax2.set_yticks(range(len(matrix_clean.index)))
    ax2.set_yticklabels(matrix_clean.index, fontsize=9)
    ax2.set_ylabel("Tần suất F (lần mua/tuần)", fontsize=10, fontweight="bold")
    ax2.set_title("B. Phân tầng F (lần/tuần) × Recency (Kiểm định F trá hình)", fontsize=11, fontweight="bold", pad=10)

    # Add text annotations in each cell
    for i in range(len(matrix_clean.index)):
        for j in range(len(matrix_clean.columns)):
            val = matrix_clean.iloc[i, j]
            text_color = "white" if val > 55 else "black"
            text_str = f"{val:.1f}%" if val > 0 else "N/A"
            ax2.text(j, i, text_str, ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax2, orientation="vertical", shrink=0.85)
    cbar.set_label("Tỷ lệ quay lại (%)", fontsize=9)

    return fig


def plot_spend_strata(spend_excess: pd.DataFrame, strata_matrix: pd.DataFrame) -> plt.Figure:
    """Plot spend/trip excess over baseline and return rate within F tiers (Q5)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # Panel 1: Excess over baseline (so với R x F nếu có)
    col = "excess_over_rf" if "excess_over_rf" in spend_excess.columns else "excess_over_baseline"
    excess_pp = spend_excess[col] * 100.0
    colors = ["#c53030" if x < 0 else "#2f855a" for x in excess_pp]
    bars = ax1.bar(
        range(len(spend_excess)),
        excess_pp,
        color=colors,
        alpha=0.85,
        width=0.5,
    )
    ax1.axhline(0, color="#4a5568", linestyle="-", linewidth=1.2)
    ax1.set_xticks(range(len(spend_excess)))
    ax1.set_xticklabels(spend_excess["spend_bin"], fontsize=9)
    ax1.set_xlabel("Phân vị Spend/trip (Q1 $\\rightarrow$ Q5)", fontsize=10, fontweight="bold")
    ylabel = "Độ chênh so với Baseline R × F (điểm %)" if col == "excess_over_rf" else "Độ chênh so với Baseline (điểm %)"
    ax1.set_ylabel(ylabel, fontsize=10, fontweight="bold")
    title1 = "A. Độ chênh tỷ lệ quay lại so với Baseline R × F" if col == "excess_over_rf" else "A. Độ chênh tỷ lệ quay lại theo Spend/trip"
    ax1.set_title(title1, fontsize=11, fontweight="bold", pad=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, excess_pp):
        offset = 0.5 if val >= 0 else -1.2
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + offset,
            f"{val:+.2f}%",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9,
            fontweight="bold",
        )

    # Panel 2: Return rate within F tiers
    f_cols = [c for c in strata_matrix.columns if c not in ("short_history", "Lịch sử ngắn (<28d)")]
    markers = ["o", "s", "^", "D", "v"]
    palette = ["#4a5568", "#3182ce", "#319795", "#d69e2e", "#e53e3e"]

    for idx, f_col in enumerate(f_cols):
        series = strata_matrix[f_col] * 100.0
        ax2.plot(
            range(len(series)),
            series.values,
            marker=markers[idx % len(markers)],
            linewidth=2,
            color=palette[idx % len(palette)],
            label=f_col,
        )

    ax2.set_xticks(range(len(strata_matrix.index)))
    ax2.set_xticklabels(strata_matrix.index, fontsize=9)
    ax2.set_xlabel("Phân vị Spend/trip", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Tỷ lệ quay lại (%)", fontsize=10, fontweight="bold")
    ax2.set_title("B. Tỷ lệ quay lại theo Spend/trip trong từng tầng F", fontsize=11, fontweight="bold", pad=10)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(title="Tầng Tần suất (F)", fontsize=8, title_fontsize=9, loc="lower right")

    return fig


def plot_ipt_band_excess(ipt_rates: pd.DataFrame) -> plt.Figure:
    """Plot return rate and excess over baseline across IPT bands (Q6)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # Panel 1: Return rate by band
    palette_rates = ["#2b6cb0", "#4299e1", "#ed8936", "#a0aec0"]
    bars1 = ax1.bar(
        range(len(ipt_rates)),
        ipt_rates["return_rate"] * 100.0,
        color=palette_rates,
        alpha=0.9,
        width=0.5,
    )
    ax1.set_xticks(range(len(ipt_rates)))
    ax1.set_xticklabels(ipt_rates["ipt_band_c"], fontsize=10, fontweight="bold")
    ax1.set_ylabel("Tỷ lệ quay lại trong 7 ngày (%)", fontsize=10, fontweight="bold")
    ax1.set_title("A. Tỷ lệ quay lại theo Nhóm Nhịp mua (IPT Band)", fontsize=11, fontweight="bold", pad=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.set_ylim(0, 85)

    for bar, (_, row) in zip(bars1, ipt_rates.iterrows()):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 2.0,
            f"{h:.1f}%\n(n={int(row['n']):,})",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    # Panel 2: Excess over baseline (so với R x F nếu có)
    col2 = "excess_over_rf" if "excess_over_rf" in ipt_rates.columns else "excess_over_baseline"
    excess_pp = ipt_rates[col2] * 100.0
    colors_excess = ["#2f855a" if x >= 0 else "#c53030" for x in excess_pp]
    bars2 = ax2.bar(
        range(len(ipt_rates)),
        excess_pp,
        color=colors_excess,
        alpha=0.85,
        width=0.5,
    )
    ax2.axhline(0, color="#2d3748", linestyle="-", linewidth=1.2)
    ax2.set_xticks(range(len(ipt_rates)))
    ax2.set_xticklabels(ipt_rates["ipt_band_c"], fontsize=10, fontweight="bold")
    ylabel2 = "Độ chênh so với Baseline R × F (điểm %)" if col2 == "excess_over_rf" else "Độ chênh so với Baseline (điểm %)"
    ax2.set_ylabel(ylabel2, fontsize=10, fontweight="bold")
    title2 = "B. Độ chênh lệch so với Baseline R × F" if col2 == "excess_over_rf" else "B. Độ chênh lệch so với Baseline 53.1%"
    ax2.set_title(title2, fontsize=11, fontweight="bold", pad=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars2, excess_pp):
        offset = 1.5 if val >= 0 else -3.5
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + offset,
            f"{val:+.1f}%",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9.5,
            fontweight="bold",
        )

    return fig

"""Visualizations for demographic coverage, attribute distributions, and excess analysis."""

from __future__ import annotations

from typing import Mapping
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from propensity.viz.style import (
    ACCENT,
    BAR,
    BOX_BG,
    FIGSIZE_FULL,
    FIGSIZE_HALF,
    LEGEND_BG,
    MEDIAN,
    PERCENTILE,
    PRIMARY,
    REF,
    TITLE,
    add_ref_line,
    annotate_box,
    fmt_pct,
    set_style,
)


def plot_demographic_coverage(profile_or_panel: pd.DataFrame) -> plt.Figure:
    """Plot dual-bar chart comparing F_bin distribution for households with vs without demographic.

    Demonstrates selection bias (groups with demographic heavily skew towards frequent shoppers).

    Parameters
    ----------
    profile_or_panel : pd.DataFrame
        Either the panel with has_demographic and F_bin, or the output of demographic_coverage_profile.

    Returns
    -------
    plt.Figure
    """
    set_style()

    f_col = "F_bin" if "F_bin" in profile_or_panel.columns else "f_bin"

    if f_col in profile_or_panel.columns and "has_demographic" in profile_or_panel.columns:
        # Cross-tabulate % distribution
        crosstab = (
            pd.crosstab(
                profile_or_panel[f_col],
                profile_or_panel["has_demographic"],
                normalize="columns",
            )
            * 100.0
        )
        # Ensure canonical category order
        if isinstance(profile_or_panel[f_col].dtype, pd.CategoricalDtype):
            cats = [c for c in profile_or_panel[f_col].cat.categories if c in crosstab.index]
            crosstab = crosstab.reindex(cats)
    else:
        raise ValueError("Dữ liệu đầu vào phải có cột 'has_demographic' và 'F_bin'")

    fig, ax = plt.subplots(figsize=FIGSIZE_FULL, dpi=150)

    n_bins = len(crosstab)
    x = np.arange(n_bins)
    bar_width = 0.36

    pct_with_demo = crosstab[1].values
    pct_no_demo = crosstab[0].values

    bars1 = ax.bar(
        x - bar_width / 2.0,
        pct_with_demo,
        width=bar_width,
        color=PRIMARY,
        edgecolor="white",
        linewidth=1.2,
        alpha=0.90,
        label="Có demographic (801 hộ, 32.1% dòng)",
        zorder=3,
    )

    bars2 = ax.bar(
        x + bar_width / 2.0,
        pct_no_demo,
        width=bar_width,
        color=BAR,
        edgecolor="white",
        linewidth=1.2,
        alpha=0.85,
        label="Không có demographic (1,699 hộ, 67.9% dòng)",
        zorder=3,
    )

    # Annotate percentages on top of bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.8,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=PRIMARY,
        )

    for bar in bars2:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.8,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=TITLE,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(crosstab.index.astype(str), fontsize=9.0)
    ax.set_xlabel("Phân tầng tần suất mua hàng (F: Ngày mua / Tuần)", fontsize=9.5, fontweight="bold", labelpad=8)
    ax.set_ylabel("Tỷ trọng trong nhóm (%)", fontsize=9.5, fontweight="bold")

    max_val = max(pct_with_demo.max(), pct_no_demo.max())
    ax.set_ylim(0, max_val * 1.25)
    fmt_pct(ax, axis="y")

    annotate_box(
        ax,
        "Selection Bias nghiêm trọng:\n"
        "• Nhóm F cao (>1.52): chiếm 38.1% ở nhóm có demo vs chỉ 11.4% ở nhóm không có\n"
        "• Nhóm F thấp (≤0.30): chỉ chiếm 3.0% ở nhóm có demo vs 28.0% ở nhóm không có\n"
        "→ Khách có demographic chủ yếu là khách thân thiết ghé thường xuyên",
        loc="upper right",
        coords=(0.97, 0.92),
    )

    title = "Q9a: Nhóm Có Dữ Liệu Nhân Khẩu Học Thiên Lệch Nặng Về Khách Mua Thường Xuyên (Selection Bias)"
    ax.set_title(title, fontsize=12, fontweight="bold", color=TITLE, pad=12)
    ax.legend(loc="upper left")

    fig.tight_layout()
    return fig


def plot_attribute_distribution(demo_grouped: pd.DataFrame, attr: str) -> plt.Figure:
    """Plot distribution of households across categories for a single demographic attribute.

    Ordered attributes (age, income, hh_size, kids) preserve their natural order.
    Unordered attributes (homeowner, marital, hh_comp) sort descending by household count.

    Parameters
    ----------
    demo_grouped : pd.DataFrame
        Output of group_demographic (801 households).
    attr : str
        Attribute name (e.g. 'age', 'income', 'homeowner', etc.).

    Returns
    -------
    plt.Figure
    """
    set_style()

    col = f"demo_{attr}" if not attr.startswith("demo_") else attr
    clean_attr = attr.replace("demo_", "")

    if col not in demo_grouped.columns:
        raise ValueError(f"Không tìm thấy cột '{col}' trong demo_grouped")

    attr_labels = {
        "age": "Độ tuổi (Age)",
        "income": "Thu nhập (Income)",
        "homeowner": "Tình trạng nhà ở (Homeowner)",
        "marital": "Tình trạng hôn nhân (Marital)",
        "hh_comp": "Cơ cấu hộ gia đình (HH Comp)",
        "hh_size": "Quy mô hộ gia đình (HH Size)",
        "kids": "Số lượng trẻ em (Kids)",
    }
    attr_name = attr_labels.get(clean_attr, clean_attr)

    counts = demo_grouped[col].value_counts(sort=False)
    total_hh = len(demo_grouped)

    # Reorder if unordered
    unordered_attrs = {"homeowner", "marital", "hh_comp"}
    if clean_attr in unordered_attrs:
        counts = counts.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=FIGSIZE_HALF, dpi=150)

    n_cats = len(counts)
    x = np.arange(n_cats)
    bar_width = 0.52

    vals = counts.values
    pcts = (vals / total_hh) * 100.0

    bars = ax.bar(
        x,
        vals,
        width=bar_width,
        color=BAR,
        edgecolor="white",
        linewidth=1.2,
        alpha=0.85,
        zorder=3,
    )

    for bar, n, p in zip(bars, vals, pcts):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + max(vals) * 0.02,
            f"{n} hộ\n({p:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color=TITLE,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(counts.index.astype(str), fontsize=8.5, rotation=15 if n_cats > 4 else 0)
    ax.set_xlabel(f"Phân loại {attr_name}", fontsize=9.0, fontweight="bold", labelpad=6)
    ax.set_ylabel("Số lượng hộ gia đình", fontsize=9.0, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.25)

    ax.set_title(f"Q9b: Phân Bổ Nhân Khẩu Học: {attr_name} (n={total_hh} hộ)", fontsize=11, fontweight="bold", color=TITLE, pad=10)

    annotate_box(ax, f"Tổng: {total_hh} hộ\nChiếm 32.0% mẫu", loc="upper right")

    fig.tight_layout()
    return fig


def plot_attribute_excess(table: pd.DataFrame) -> plt.Figure:
    """Plot summary dot plot comparing excess return rates over R x F across all demographic attributes.

    Displays shaded neutral zone (+/- 3 pp) and dots labeled with household sample sizes.

    Parameters
    ----------
    table : pd.DataFrame
        Output of attribute_excess_table across all attributes.

    Returns
    -------
    plt.Figure
    """
    set_style()
    df = table.copy()

    # Create composite row label "attribute: level"
    df["row_label"] = df["attribute"] + " — " + df["level"]

    # Reverse rows so first attribute appears at top
    df = df.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10.5, 9.5), dpi=150)

    y_pos = np.arange(len(df))

    # Neutral zone shading (+/- 3 pp)
    ax.axvspan(-3.0, 3.0, color="#E2E8F0", alpha=0.50, label="Vùng không có thêm tín hiệu ngoài R × F (±3 pp)", zorder=1)
    add_ref_line(ax, 0.0, label="Mốc 0 (Kỳ vọng R × F)", is_vertical=True)

    # Draw horizontal separator lines between different attributes
    current_attr = df["attribute"].iloc[0]
    for i, attr in enumerate(df["attribute"]):
        if attr != current_attr:
            ax.axhline(i - 0.5, color="#CBD5E1", linestyle="--", linewidth=0.8, alpha=0.8, zorder=2)
            current_attr = attr

    # Plot points
    excess_vals = df["excess_pp"].values
    n_households = df["n_households"].values

    dot_colors = []
    has_pos_insight = False
    has_neg_insight = False
    has_neutral = False

    for _, row in df.iterrows():
        v = str(row.get("verdict", ""))
        ex = float(row.get("excess_pp", 0.0))
        if "Insight" in v:
            if ex >= 0:
                dot_colors.append(PRIMARY)
                has_pos_insight = True
            else:
                dot_colors.append(ACCENT)
                has_neg_insight = True
        else:
            dot_colors.append(BAR)
            has_neutral = True

    # Plot scatter dots
    ax.scatter(
        excess_vals,
        y_pos,
        color=dot_colors,
        edgecolors="white",
        linewidths=1.2,
        s=90,
        zorder=5,
    )

    # Annotate each dot with excess_pp and n_households
    for y, ex, n_hh in zip(y_pos, excess_vals, n_households):
        sign = "+" if ex > 0 else ""
        txt = f" {sign}{ex:.1f} pp (n={int(n_hh):,})"
        ha = "left" if ex >= 0 else "right"
        offset_x = 0.2 if ex >= 0 else -0.2
        ax.text(
            ex + offset_x,
            y,
            txt,
            va="center",
            ha=ha,
            fontsize=8.0,
            fontweight="semibold",
            color=TITLE,
            zorder=6,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["row_label"], fontsize=8.5, fontweight="medium")
    ax.set_xlabel("Chênh lệch tỷ lệ quay lại so với kỳ vọng R × F (Excess pp)", fontsize=9.5, fontweight="bold", labelpad=8)
    ax.set_ylabel("Thuộc tính & Phân nhóm nhân khẩu học", fontsize=9.5, fontweight="bold")
    ax.set_xlim(-6.0, 8.5)

    # Legend
    legend_handles = [
        mpatches.Patch(color="#E2E8F0", label="Vùng không phân hóa (±3 pp)"),
        mpl.lines.Line2D([0], [0], color=REF, linestyle="--", linewidth=1.4, label="Kỳ vọng R × F (0 pp)"),
    ]
    if has_pos_insight:
        legend_handles.append(mpl.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=PRIMARY, markersize=8, label="Nhóm có demographic chung (+4.6 pp lift)"))
    if has_neutral:
        legend_handles.append(mpl.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=BAR, markersize=8, label="Không ổn định / Không vượt trội"))

    ax.legend(handles=legend_handles, loc="upper right", frameon=True, facecolor=LEGEND_BG, fontsize=8.5)

    title = "Q9c: Sau Khi Kiểm Soát R × F, Toàn Bộ 7 Thuộc Tính Nhân Khẩu Học Biến Thiên Đồng Nhất (+4.0 đến +5.5 pp)"
    ax.set_title(title, fontsize=12, fontweight="bold", color=TITLE, pad=12)

    annotate_box(
        ax,
        "Đồng nhất cao độ:\n"
        "• Toàn bộ 26 phân nhóm đều nằm sát mức +4.6 pp (thừa hưởng từ selection bias)\n"
        "• Độ tuổi, thu nhập, số con... không tạo ra khác biệt >1.5 pp về tỷ lệ quay lại\n"
        "→ Hành vi mua sắm (R × F) quyết định việc quay lại, nhân khẩu học không bổ sung thêm thông tin",
        loc="lower left",
        coords=(0.02, 0.02),
        fontsize=8.0,
    )

    fig.tight_layout()
    return fig

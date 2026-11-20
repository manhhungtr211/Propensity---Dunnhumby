"""Visualization functions for promotion and marketing campaign analytics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from propensity.viz.style import (
    BG_COLOR,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    CMAP_SEQUENTIAL_BLUE,
    GRAY_LIGHT,
    GRAY_MUTED,
    GRAY_NEUTRAL,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    set_style,
)


def plot_weekly_promo(
    weekly: pd.DataFrame,
    campaign_summary: pd.DataFrame | None = None,
) -> plt.Figure:
    """Plot weekly promotion dynamics using small multiples with shared x-axis (tuần).

    Upper panel: promo_spend_share (%) and discount_depth (%).
    Lower panel: total merchandise sales ($).
    Background shading: Type A campaign timeframes across both panels.
    Strictly avoids dual y-axes.
    """
    set_style()
    df = weekly.sort_values("WEEK_NO").reset_index(drop=True)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 7.5), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [1.4, 1.0]}
    )

    weeks = df["WEEK_NO"].to_numpy()

    # Identify Type A campaigns for background shading
    type_a_spans: list[tuple[float, float]] = []
    if campaign_summary is not None and not campaign_summary.empty:
        type_a = campaign_summary[campaign_summary["type"].astype(str).str.contains("A|TypeA|Type A", case=False)]
        for _, row in type_a.iterrows():
            w_start = (row["start"] - 1) / 7.0 + 1.0
            w_end = (row["end"] - 1) / 7.0 + 1.0
            type_a_spans.append((w_start, w_end))

    # Apply shading to both panels
    for i, (ws, we) in enumerate(type_a_spans):
        lbl = "Giai đoạn Campaign Loại A" if i == 0 else None
        ax_top.axvspan(ws, we, color=GRAY_NEUTRAL, alpha=0.85, label=lbl, zorder=1)
        ax_bot.axvspan(ws, we, color=GRAY_NEUTRAL, alpha=0.85, zorder=1)

    # Upper panel: Promo Spend Share & Discount Depth
    ax_top.plot(
        weeks, df["promo_spend_share"] * 100.0,
        color=COLOR_PRIMARY, linewidth=2.0, label="Tỷ trọng chi tiêu hàng giảm giá (Promo spend share %)", zorder=3
    )
    ax_top.plot(
        weeks, df["discount_depth"] * 100.0,
        color=COLOR_SECONDARY, linewidth=1.8, linestyle="-", label="Độ sâu giảm giá (Discount depth % trên Gross)", zorder=3
    )

    ax_top.set_ylabel("Tỷ lệ (%)", fontsize=10, fontweight="bold")
    ax_top.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    ax_top.set_title(
        "Diễn Biến Giảm Giá & Doanh Số Theo Tuần (Q6: Weekly Promotion & Campaign Overlap)\n"
        "[Nền xám: Các chiến dịch tiếp thị trọng điểm Loại A]",
        fontsize=12, fontweight="bold", pad=10
    )
    ax_top.legend(frameon=True, facecolor=BG_COLOR, edgecolor=GRAY_LIGHT, loc="upper right", fontsize=8.5)

    # Lower panel: Sales
    ax_bot.bar(
        weeks, df["sales"] / 1000.0,
        color=GRAY_MUTED, alpha=0.85, width=0.8, label="Doanh số hàng hóa (Nghìn $)", zorder=3
    )
    ax_bot.set_xlabel("Tuần trong năm (WEEK_NO)", fontsize=10, fontweight="bold", labelpad=6)
    ax_bot.set_ylabel("Doanh số (Nghìn $)", fontsize=10, fontweight="bold")
    ax_bot.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}k"))
    ax_bot.legend(frameon=True, facecolor=BG_COLOR, edgecolor=GRAY_LIGHT, loc="upper right", fontsize=8.5)

    ax_top.set_xlim(float(weeks.min()) - 1, float(weeks.max()) + 1)
    return fig


def plot_department_promo(dept: pd.DataFrame) -> plt.Figure:
    """Plot horizontal ranked bar chart of promotion spend share by department with spend share annotations."""
    set_style()
    df = dept.sort_values("promo_spend_share", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(5.0, len(df) * 0.42)), constrained_layout=True)
    y_pos = np.arange(len(df))

    bars = ax.barh(y_pos, df["promo_spend_share"] * 100.0, color=COLOR_PRIMARY, height=0.65, zorder=3)

    # Annotations: Promo % and Spend share %
    for bar, (_, row) in zip(bars, df.iterrows()):
        val = row["promo_spend_share"] * 100.0
        sp_share = row["spend_share"] * 100.0
        ax.text(
            val + 0.8, bar.get_y() + bar.get_height() / 2.0,
            f"{val:.1f}% (Chiếm {sp_share:.1f}% doanh số)",
            va="center", ha="left", fontsize=8.5, fontweight="semibold", color=TEXT_PRIMARY
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["department"], fontsize=9.5, fontweight="semibold")
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    ax.set_xlabel("Tỷ trọng chi tiêu hàng có giảm giá (Promo Spend Share %)", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_title(
        "Mức Độ Phụ Thuộc Giảm Giá Theo Ngành Hàng (Q6: Department Promotion Share)\n"
        "[Sắp xếp giảm dần theo mức độ săn đón khuyến mãi]",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlim(0, max(60.0, float(df["promo_spend_share"].max() * 100.0) + 18.0))
    return fig


def plot_promo_dept_week(matrix: pd.DataFrame) -> plt.Figure:
    """Plot heatmap of promo_spend_share for top departments across weeks using sequential blue palette."""
    set_style()
    fig, ax = plt.subplots(figsize=(12, max(4.5, len(matrix) * 0.5)), constrained_layout=True)

    data = matrix.to_numpy() * 100.0
    im = ax.imshow(data, aspect="auto", cmap=CMAP_SEQUENTIAL_BLUE, origin="upper", vmin=0.0, vmax=100.0)

    ax.set_yticks(np.arange(len(matrix)))
    ax.set_yticklabels(matrix.index, fontsize=9.5, fontweight="semibold")

    # X ticks every 10 weeks
    cols = [int(c) for c in matrix.columns]
    step = 10
    tick_locs = [i for i, c in enumerate(cols) if c % step == 0 or i == 0 or i == len(cols) - 1]
    ax.set_xticks(tick_locs)
    ax.set_xticklabels([str(cols[i]) for i in tick_locs], fontsize=9)

    ax.set_xlabel("Tuần trong năm (WEEK_NO)", fontsize=10, fontweight="bold", labelpad=6)
    ax.set_title(
        "Ma Trận Giảm Giá Theo Ngành Hàng & Tuần (Q6: Department × Week Promo Heatmap)\n"
        "[% Doanh số bán ra có giảm giá giá niêm yết (RETAIL_DISC < 0)]",
        fontsize=12, fontweight="bold", pad=12
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label("Tỷ lệ giảm giá (%)", fontsize=9.5, fontweight="bold")
    cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))

    return fig


def plot_campaign_overview(
    summary: pd.DataFrame,
    type_summary: pd.DataFrame,
) -> plt.Figure:
    """Plot campaign timeline (Panel A) and redemption rate by type (Panel B)."""
    set_style()
    df = summary.sort_values("start", ascending=True).reset_index(drop=True)

    fig, (ax_time, ax_rate) = plt.subplots(
        1, 2, figsize=(13, 6.0), constrained_layout=True,
        gridspec_kw={"width_ratios": [2.2, 1.0]}
    )

    # Color mapping for types
    type_colors = {
        "TypeA": COLOR_PRIMARY,
        "Type A": COLOR_PRIMARY,
        "TypeB": COLOR_SECONDARY,
        "Type B": COLOR_SECONDARY,
        "TypeC": COLOR_TERTIARY,
        "Type C": COLOR_TERTIARY,
    }

    # Panel A: Timeline
    y_pos = np.arange(len(df))
    for y, (_, row) in zip(y_pos, df.iterrows()):
        t = str(row["type"]).strip()
        c = type_colors.get(t, GRAY_MUTED)
        start = row["start"]
        dur = row["duration"]
        ax_time.barh(y, dur, left=start, height=0.6, color=c, zorder=3)
        # Direct label household count
        n_hh = row["n_hh_received"]
        ax_time.text(
            start + dur + 4, y, f"C{row['CAMPAIGN']} ({n_hh:,} hộ)",
            va="center", ha="left", fontsize=7.5, color=TEXT_SECONDARY
        )

    ax_time.set_yticks(y_pos)
    ax_time.set_yticklabels([f"C{r['CAMPAIGN']} ({r['type']})" for _, r in df.iterrows()], fontsize=8)
    ax_time.set_xlabel("Ngày trong năm (DAY)", fontsize=10, fontweight="bold")
    ax_time.set_title(
        "Panel A: Dòng Thời Gian Các Chiến Dịch (Campaign Timelines)\n"
        "[Start Day → End Day; Nhãn: Mã campaign & số hộ tiếp cận]",
        fontsize=11, fontweight="bold", pad=8
    )

    # Legend for types
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_PRIMARY, label="Loại A (Đại trà / Trọng điểm)"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_SECONDARY, label="Loại B (Mục tiêu nhóm)"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_TERTIARY, label="Loại C (Chuyên biệt)"),
    ]
    ax_time.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=True, facecolor=BG_COLOR, edgecolor=GRAY_LIGHT)

    # Panel B: Redemption Rate by Type
    ts = type_summary.sort_values("type").reset_index(drop=True)
    y_ts = np.arange(len(ts))
    b_colors = [type_colors.get(str(t).strip(), GRAY_MUTED) for t in ts["type"]]

    bars = ax_rate.barh(y_ts, ts["redemption_rate"] * 100.0, color=b_colors, height=0.55, zorder=3)
    for bar, (_, row) in zip(bars, ts.iterrows()):
        val = row["redemption_rate"] * 100.0
        n_red = row["total_hh_redeemed"]
        n_rec = row["total_hh_received"]
        ax_rate.text(
            val + 0.6, bar.get_y() + bar.get_height() / 2.0,
            f"{val:.1f}%\n({n_red:,}/{n_rec:,} lượt)",
            va="center", ha="left", fontsize=8.5, fontweight="bold", color=TEXT_PRIMARY
        )

    ax_rate.set_yticks(y_ts)
    ax_rate.set_yticklabels(ts["type"], fontsize=10, fontweight="semibold")
    ax_rate.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    ax_rate.set_xlabel("Tỷ lệ đổi coupon (Redemption Rate %)", fontsize=10, fontweight="bold")
    ax_rate.set_title(
        "Panel B: Tỷ Lệ Đổi Coupon Gộp\n"
        "[Theo từng loại chiến dịch]",
        fontsize=11, fontweight="bold", pad=8
    )
    ax_rate.set_xlim(0, max(25.0, float(ts["redemption_rate"].max() * 100.0) + 10.0))

    return fig


def plot_campaign_reach(reach: dict[str, pd.DataFrame]) -> plt.Figure:
    """Plot selection bias: share of panel observations in active campaign across F_bin and recency_bin."""
    set_style()
    fig, (ax_f, ax_r) = plt.subplots(1, 2, figsize=(12, 5.0), constrained_layout=True)

    # Panel 1: By Frequency bin
    df_f = reach["by_f"]
    y_f = np.arange(len(df_f))
    bars_f = ax_f.barh(y_f, df_f["reach_rate"] * 100.0, color=COLOR_PRIMARY, height=0.6, zorder=3)
    for bar, (_, row) in zip(bars_f, df_f.iterrows()):
        val = row["reach_rate"] * 100.0
        ax_f.text(
            val + 0.8, bar.get_y() + bar.get_height() / 2.0,
            f"{val:.1f}%", va="center", ha="left", fontsize=9, fontweight="bold", color=TEXT_PRIMARY
        )

    ax_f.set_yticks(y_f)
    col_name_f = df_f.columns[0]
    ax_f.set_yticklabels(df_f[col_name_f], fontsize=9.5, fontweight="semibold")
    ax_f.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    ax_f.set_xlabel("% Quan sát đang trong chiến dịch", fontsize=9.5, fontweight="bold")
    ax_f.set_title(
        "Độ Phủ Chiến Dịch Theo Tần Suất F\n"
        "[F càng cao, xác suất nhận chiến dịch càng lớn]",
        fontsize=10.5, fontweight="bold", pad=8
    )
    ax_f.set_xlim(0, max(60.0, float(df_f["reach_rate"].max() * 100.0) + 12.0))

    # Panel 2: By Recency bin
    df_r = reach["by_r"]
    y_r = np.arange(len(df_r))
    bars_r = ax_r.barh(y_r, df_r["reach_rate"] * 100.0, color=COLOR_SECONDARY, height=0.6, zorder=3)
    for bar, (_, row) in zip(bars_r, df_r.iterrows()):
        val = row["reach_rate"] * 100.0
        ax_r.text(
            val + 0.8, bar.get_y() + bar.get_height() / 2.0,
            f"{val:.1f}%", va="center", ha="left", fontsize=9, fontweight="bold", color=TEXT_PRIMARY
        )

    ax_r.set_yticks(y_r)
    col_name_r = df_r.columns[0]
    ax_r.set_yticklabels(df_r[col_name_r], fontsize=9.5, fontweight="semibold")
    ax_r.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    ax_r.set_xlabel("% Quan sát đang trong chiến dịch", fontsize=9.5, fontweight="bold")
    ax_r.set_title(
        "Độ Phủ Chiến Dịch Theo Recency R\n"
        "[Mua càng gần, xác suất nhận chiến dịch càng lớn]",
        fontsize=10.5, fontweight="bold", pad=8
    )
    ax_r.set_xlim(0, max(60.0, float(df_r["reach_rate"].max() * 100.0) + 12.0))

    return fig

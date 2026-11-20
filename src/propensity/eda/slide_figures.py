"""Slide-ready figure generation conforming to Storytelling with Data and presentation rules.

Rules applied across all 13 slide figures:
- 1 idea per slide, headlines placed on Canva.
- Values on bars -> hide value axis and grid (hide_value_axis).
- Monochromatic / Selective Focus: GRAY_BAR for context, RED for slide focal point.
- Direct labeling replaces legend boxes.
- Linear scale 1–30 for S5 cumulative with red shaded area under <=7 days.
- Vietnamese number formatting: decimal comma (53,1%), thousands dot (2.500).
- Single axis per figure, zero twinx, zero suptitle.
"""

from __future__ import annotations

from typing import Any, Final, Sequence
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from propensity.viz.slide_style import (
    BAND,
    GRAY_BAR,
    GRAY_TEXT,
    INK,
    NAVY,
    RED,
    RED_LIGHT,
    SLIDE_FULL,
    SLIDE_HALF,
    annotate_point,
    direct_label,
    fmt_money,
    fmt_num,
    fmt_pct,
    hide_value_axis,
    pct_axis,
    set_slide_style,
    thousands_axis,
)


def table_coverage(
    tx: pd.DataFrame,
    campaign_table: pd.DataFrame,
    coupon_redempt: pd.DataFrame,
    hh_demographic: pd.DataFrame,
    total_hh: int = 2500,
) -> pd.DataFrame:
    """Calculate household coverage count and percentage across core and secondary tables."""
    def _get_hh_count(df: pd.DataFrame) -> int:
        for c in ["household_key", "HOUSEHOLD_KEY"]:
            if c in df.columns:
                return int(df[c].nunique())
        for c in df.columns:
            if c.lower() == "household_key":
                return int(df[c].nunique())
        raise ValueError("Không tìm thấy cột household_key trong bảng")

    rows = [
        {"table_name": "Giao dịch (Transaction)", "n_households": _get_hh_count(tx)},
        {"table_name": "Chiến dịch (Campaign)", "n_households": _get_hh_count(campaign_table)},
        {"table_name": "Nhân khẩu học (Demographic)", "n_households": _get_hh_count(hh_demographic)},
        {"table_name": "Coupon đã đổi (Coupon redempt)", "n_households": _get_hh_count(coupon_redempt)},
    ]
    res = pd.DataFrame(rows)
    res["pct_households"] = (res["n_households"] / total_hh) * 100.0
    return res.sort_values("pct_households", ascending=False).reset_index(drop=True)


def ipt_gap_distribution(
    purchase_days: pd.DataFrame | pd.Series | np.ndarray,
) -> pd.Series:
    """Extract consecutive purchase day gaps (gap >= 1) per household."""
    if isinstance(purchase_days, pd.Series):
        gaps = purchase_days.dropna()
    elif isinstance(purchase_days, np.ndarray):
        gaps = pd.Series(purchase_days).dropna()
    elif isinstance(purchase_days, pd.DataFrame):
        if "gap" in purchase_days.columns:
            gaps = purchase_days["gap"].dropna()
        elif "household_key" in purchase_days.columns and "DAY" in purchase_days.columns:
            sub = (
                purchase_days[["household_key", "DAY"]]
                .drop_duplicates()
                .sort_values(["household_key", "DAY"])
            )
            gaps = sub.groupby("household_key")["DAY"].diff().dropna()
        else:
            raise ValueError("DataFrame phải chứa cột 'gap' hoặc ('household_key', 'DAY')")
    else:
        raise ValueError(f"Định dạng không được hỗ trợ: {type(purchase_days)}")

    # Gaps between distinct purchase days must be >= 1
    gaps = gaps[gaps >= 1].astype(float).reset_index(drop=True)
    return gaps


# ==============================================================================
# Slide Figures: Core Story (V1, V2, V3)
# ==============================================================================

def fig_v1(cells: pd.DataFrame) -> plt.Figure:
    """S8_rf_heatmap: 2D R x F Baseline matrix with divergent colormap centered at 53.1%.

    Storytelling update:
    - Bỏ thanh màu (colorbar) vì trong ô đã có số trực tiếp.
    - Hộp chú thích bên ngoài heatmap, không che khuất bất kỳ hàng nào.
    """
    set_slide_style()
    fig, ax = plt.subplots(figsize=SLIDE_FULL)

    # Filter out short history and ensure standard order
    df = cells[cells["f_bin"].astype(str) != "Lịch sử ngắn (<28d)"].copy()
    r_order = ["1–7", "8–14", "15–30", "31+"]
    f_order = ["≤ 0,30", "0,30–0,57", "0,57–0,96", "0,96–1,52", "> 1,52"]

    f_map = {
        "≤ 0.30": "≤ 0,30",
        "0.30–0.57": "0,30–0,57",
        "0.57–0.96": "0,57–0,96",
        "0.96–1.52": "0,96–1,52",
        "> 1.52": "> 1,52",
    }
    df["f_bin_vn"] = df["f_bin"].astype(str).map(lambda x: f_map.get(x, x))

    matrix = df.pivot(index="r_bin", columns="f_bin_vn", values="rate").reindex(
        index=r_order, columns=f_order
    )

    # Diverging colormap: RED (<53.1%) -> White (53.1%) -> NAVY/INK (>53.1%)
    cmap = mcolors.LinearSegmentedColormap.from_list("rf_divergent", [RED, "#FFFFFF", "#1E3A5F"])
    norm = mcolors.TwoSlopeNorm(vcenter=0.5311, vmin=0.08, vmax=0.95)

    im = ax.imshow(matrix.values, cmap=cmap, norm=norm, aspect="auto")

    # Cell text annotations with decimal comma
    for r_idx in range(len(r_order)):
        for f_idx in range(len(f_order)):
            val = matrix.iloc[r_idx, f_idx]
            if np.isnan(val):
                continue
            text_color = "#FFFFFF" if (val < 0.25 or val > 0.75) else INK
            ax.text(
                f_idx,
                r_idx,
                fmt_pct(val, 1, is_ratio=True),
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color=text_color,
            )

    # Box borders around highest cell (0, 4) and lowest cell (3, 0)
    rect_high = Rectangle((4 - 0.5, 0 - 0.5), 1, 1, fill=False, edgecolor=INK, lw=2.5, zorder=5)
    ax.add_patch(rect_high)

    rect_low = Rectangle((0 - 0.5, 3 - 0.5), 1, 1, fill=False, edgecolor=INK, lw=2.5, zorder=5)
    ax.add_patch(rect_low)

    # Annotation outside the heatmap (to the right, leaving rows 15-30 fully visible)
    ax.annotate(
        "Khách từng mua đều\nnhưng đã vắng > 30 ngày\n(Chỉ còn 12,5% quay lại)",
        xy=(4, 3),
        xytext=(1.04, 0.14),
        textcoords="axes fraction",
        fontsize=12,
        fontweight="bold",
        color=INK,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5, shrinkA=3, shrinkB=6),
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="#FFFFFF", ec=RED, lw=1.2),
    )

    ax.set_xticks(range(len(f_order)))
    ax.set_xticklabels(f_order, fontsize=12)
    ax.set_yticks(range(len(r_order)))
    ax.set_yticklabels(r_order, fontsize=12)

    ax.set_xlabel("Tần suất mua (ngày/tuần)", fontsize=13, color=GRAY_TEXT, labelpad=8)
    ax.set_ylabel("Số ngày từ lần mua cuối", fontsize=13, color=GRAY_TEXT, labelpad=8)

    # Bỏ thanh màu (colorbar) theo quy tắc Storytelling: nhãn số trong ô đã đầy đủ
    ax.grid(False)
    return fig


def fig_v2(summary_df: pd.DataFrame) -> plt.Figure:
    """S9_raw_vs_rf: The 'Aha' dumbbell chart showing collapse of raw differences after R x F.

    Storytelling update:
    - Xóa hộp legend.
    - Nhãn trực tiếp 'So trực tiếp' và 'Sau khi tính mức mua gần đây & thường xuyên' ở đỉnh chart.
    - Chấm rỗng viền đỏ ruột trắng cho 'Có thông tin khảo sát' kèm ghi chú (selection bias).
    - Chữ 'Vùng không đáng kể (< 3 pp)' ở đáy dải (y = 0.2).
    """
    set_slide_style()
    fig, ax = plt.subplots(figsize=SLIDE_FULL)

    df = summary_df.sort_values("raw_pp", ascending=True).reset_index(drop=True)
    # Standardize feature names
    df["feature_vn"] = df["feature"].replace({"Có demographic": "Có thông tin khảo sát"})
    n = len(df)
    y_positions = np.arange(n)

    # Indifference zone band [0, 3] pp
    ax.axvspan(0, 3, color=BAND, alpha=0.9, zorder=1)
    ax.text(
        1.5,
        0.2,
        "Vùng không đáng kể (< 3 pp)",
        ha="center",
        va="bottom",
        fontsize=11.5,
        fontstyle="italic",
        color=RED,
        fontweight="bold",
        zorder=2,
    )

    # Connecting dumbbell lines and markers
    for i, row in df.iterrows():
        raw = row["raw_pp"]
        adj = row["adj_pp"]
        verdict = row["best_verdict"]
        feat_name = row["feature_vn"]

        # Connecting line
        ax.plot([adj, raw], [i, i], color=GRAY_BAR, lw=3.0, zorder=2)

        # Raw dot
        ax.scatter(raw, i, color=GRAY_BAR, s=110, edgecolor="white", lw=1.2, zorder=3)

        # Adjusted dot
        if feat_name == "Có thông tin khảo sát":
            ax.scatter(adj, i, facecolor="white", edgecolor=RED, lw=2.2, s=120, zorder=5)
            ax.text(
                adj + 0.45,
                i,
                "(selection bias)",
                va="center",
                ha="left",
                fontsize=10.5,
                fontstyle="italic",
                color=RED,
            )
        else:
            adj_color = RED if verdict == "Insight" else INK
            ax.scatter(adj, i, color=adj_color, s=120, edgecolor="white", lw=1.2, zorder=4)

    # Custom y-tick labels with bold on Insight
    ax.set_yticks(y_positions)
    labels = df["feature_vn"].tolist()
    ax.set_yticklabels(labels, fontsize=12)
    for idx, tick in enumerate(ax.get_yticklabels()):
        if df.iloc[idx]["best_verdict"] == "Insight" and df.iloc[idx]["feature_vn"] != "Có thông tin khảo sát":
            tick.set_fontweight("bold")
            tick.set_color(INK)
        else:
            tick.set_color(GRAY_TEXT)

    ax.set_xlabel("Độ chênh lệch lớn nhất (điểm %)", fontsize=13, color=GRAY_TEXT, labelpad=8)
    ax.set_xlim(-0.5, max(df["raw_pp"].max() + 3.0, 36.0))
    ax.set_ylim(-0.6, n + 0.8)

    # Nhãn trực tiếp trên đỉnh chart (thay thế hoàn toàn hộp legend)
    y_top = n + 0.1
    ax.scatter(0.0, y_top, color=GRAY_BAR, s=100, edgecolor="white", lw=1.2, zorder=5)
    ax.text(0.7, y_top, "So trực tiếp", va="center", ha="left", fontsize=11.5, color=GRAY_TEXT, fontweight="bold")

    ax.scatter(7.0, y_top, color=RED, s=100, edgecolor="white", lw=1.2, zorder=5)
    ax.text(7.7, y_top, "Sau khi tính mức mua gần đây & thường xuyên", va="center", ha="left", fontsize=11.5, color=RED, fontweight="bold")

    ax.scatter(27.0, y_top, facecolor="white", edgecolor=RED, lw=2.0, s=100, zorder=5)
    ax.text(27.8, y_top, "Có thông tin khảo sát (selection bias)", va="center", ha="left", fontsize=11, color=RED)

    ax.grid(axis="x", color="#E5E7EB", linestyle="-", lw=0.8)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    return fig


def fig_v3(trend_excess_table: pd.DataFrame) -> plt.Figure:
    """S10_trend_excess: Recent IPT trend ratio excess over R x F with early churn warning.

    Storytelling update:
    - Bỏ màu xanh than: chỉ 'Mua thưa hẳn' tô đỏ, 3 nhóm còn lại tô xám.
    - Xóa hộp legend. Ghi chữ 'Dự đoán' 1 lần cạnh vạch đen.
    - Bỏ trục Y và grid trục Y (nhãn giá trị đặt trong bar).
    - 4 nhãn nhóm mới: Mua dày hơn, Đều đặn, Hơi thưa, Mua thưa hẳn.
    """
    set_slide_style()
    fig, ax = plt.subplots(figsize=SLIDE_FULL)

    group_col = trend_excess_table.columns[0]
    df = trend_excess_table[
        ~trend_excess_table[group_col].astype(str).str.lower().isin(["null", "nan", "none", ""])
    ].copy()

    # Reorder cleanly with 4 new labels
    label_map = {
        "< 0.75 (Tăng tốc)": "Mua dày hơn",
        "0.75–1.25 (Ổn định)": "Đều đặn",
        "1.25–2.0 (Chậm lại)": "Hơi thưa",
        "> 2.0 (Chậm hẳn)": "Mua thưa hẳn",
    }
    df["label_vn"] = df[group_col].map(lambda x: label_map.get(str(x), str(x)))
    order_map = {"Mua dày hơn": 0, "Đều đặn": 1, "Hơi thưa": 2, "Mua thưa hẳn": 3}
    df["order"] = df["label_vn"].map(order_map).fillna(99)
    df = df.sort_values("order").reset_index(drop=True)

    x = np.arange(len(df))
    width = 0.52

    # Monochromatic palette: only index 3 ("Mua thưa hẳn") is RED, others GRAY_BAR
    bar_colors = [RED if i == 3 else GRAY_BAR for i in range(len(df))]
    bars = ax.bar(x, df["observed"] * 100.0, width=width, color=bar_colors, edgecolor="white", lw=1.5, zorder=2)

    # Expected horizontal line on each bar and labels INSIDE the bars
    for i, row in df.iterrows():
        exp_pct = row["expected"] * 100.0
        obs_pct = row["observed"] * 100.0
        exc = row["excess_pp"]

        # Expected horizontal tick line
        ax.plot([i - width / 2.0, i + width / 2.0], [exp_pct, exp_pct], color=INK, lw=3.2, zorder=4)

        # Place label INSIDE bar: white text for RED, dark INK text for GRAY_BAR
        text_color = "#FFFFFF" if bar_colors[i] == RED else INK
        sign_str = "+" if exc > 0 else ""
        y_text = obs_pct - 14.0 if obs_pct > 30 else obs_pct / 2.0

        ax.text(
            i,
            y_text,
            f"{fmt_pct(obs_pct, 1)}\n({sign_str}{fmt_pct(exc, 1).replace('%', ' pp')})",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=text_color,
            zorder=5,
        )

    # Ghi chữ 'Dự đoán' một lần cạnh vạch đen ở bar đầu tiên
    exp_0 = df.iloc[0]["expected"] * 100.0
    ax.text(
        0,
        exp_0 + 2.0,
        "Dự đoán",
        ha="center",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
        zorder=5,
    )

    # X-axis ticks with household count
    x_labels = []
    for _, row in df.iterrows():
        name = str(row["label_vn"])
        n_hh = int(row["n_households"])
        x_labels.append(f"{name}\n(n = {fmt_num(n_hh)} hộ)")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_ylim(0, 85)

    # Bỏ trục Y và grid trục Y theo quy tắc nhãn số trực tiếp
    hide_value_axis(ax, axis="y")

    # Annotate "Mua thưa hẳn" in the upper open space (y = 74)
    ax.text(
        2.6,
        74.0,
        "Thấp hơn dự đoán 7,3 pp ở 100% thời điểm",
        ha="center",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
        color=RED,
    )

    ax.set_axisbelow(True)
    return fig


# ==============================================================================
# Slide Figures: Label & Cutoff Stability (L1, L2a, L2b, L3)
# ==============================================================================

def fig_l1(weekly_df: pd.DataFrame) -> plt.Figure:
    """S4_weekly_active: Weekly active households stabilizing after Day 140."""
    set_slide_style()
    fig, ax = plt.subplots(figsize=SLIDE_FULL)

    # Exclude partial weeks (week 1 and last week < 7 days)
    df = weekly_df.copy()
    if len(df) > 2:
        df = df[(df["WEEK_NO"] > 1) & (df["WEEK_NO"] < df["WEEK_NO"].max())].reset_index(drop=True)

    days = df["WEEK_NO"] * 7
    active = df["active_hh"]

    # Shaded warmup period [0, 140]
    ax.axvspan(0, 140, color=GRAY_BAR, alpha=0.35, zorder=1)
    ax.text(
        70,
        1500,
        "Giai đoạn khởi động",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=GRAY_TEXT,
        zorder=2,
    )

    # Weekly active line
    ax.plot(days, active, color=RED, lw=2.5, zorder=3)

    # Vertical line at cutoff 140
    ax.axvline(140, color=INK, linestyle="--", lw=1.5, zorder=4)
    ax.text(
        145,
        1500,
        "Cutoff đầu tiên (Ngày 140)",
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )

    # Mean line after Day 140
    post_140 = df[df["min_day"] >= 140]
    mean_val = float(post_140["active_hh"].mean()) if not post_140.empty else 1322.0

    ax.axhline(mean_val, color=GRAY_TEXT, linestyle="--", lw=1.2, zorder=2)
    ax.text(
        days.iloc[-1] - 10,
        1500,
        f"Trung bình sau ngày 140: {fmt_num(mean_val)} hộ/tuần",
        ha="right",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
    )

    ax.set_xlabel("Ngày quan sát (DAY)", fontsize=13, color=GRAY_TEXT)
    ax.set_ylabel("Số hộ hoạt động / tuần", fontsize=13, color=GRAY_TEXT)
    ax.set_xlim(0, days.iloc[-1] + 14)
    ax.set_ylim(0, 1680)
    thousands_axis(ax, "y")
    ax.set_axisbelow(True)

    return fig


def fig_l2a(gaps: pd.Series) -> plt.Figure:
    """S5_ipt_hist: Backup histogram 1–30 days with >30 tail and cumulative notes at 3, 7, 30 days.

    Storytelling update:
    - Cắt đuôi: 1 đến 30, gộp >30 thành cột cuối.
    - Cột 1–7 tô đỏ, cột 8+ tô xám.
    - % tích lũy trên đỉnh cột tại mốc 3, 7, 30 ngày.
    - Bỏ trục Y vì đã có nhãn.
    """
    set_slide_style()
    fig, ax = plt.subplots(figsize=SLIDE_HALF)

    clean_gaps = ipt_gap_distribution(gaps)
    total_gaps = len(clean_gaps)

    # Bins 1..30 and >30
    counts = [float((clean_gaps == i).sum()) for i in range(1, 31)]
    count_over_30 = float((clean_gaps > 30).sum())
    all_counts = counts + [count_over_30]
    shares = [(c / total_gaps) * 100.0 for c in all_counts]

    x_vals = np.arange(1, 32)
    # 1..7 RED, 8..31 GRAY_BAR
    bar_colors = [RED if i <= 7 else GRAY_BAR for i in x_vals]

    ax.bar(x_vals, shares, color=bar_colors, width=0.85, edgecolor="white", lw=0.5, zorder=2)

    # Cumulative % notes at 3, 7, and 30 days
    pct_3 = float((clean_gaps <= 3).mean() * 100.0)
    pct_7 = float((clean_gaps <= 7).mean() * 100.0)
    pct_30 = float((clean_gaps <= 30).mean() * 100.0)

    # Label on bar 3
    ax.text(3, shares[2] + 0.5, f"Tích lũy 3d:\n{fmt_pct(pct_3, 1)}", ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
    # Label on bar 7
    ax.text(7, shares[6] + 0.5, f"Tích lũy 7d:\n{fmt_pct(pct_7, 1)}", ha="center", va="bottom", fontsize=10.5, color=RED, fontweight="bold")
    # Label on bar 30
    ax.text(30, shares[29] + 0.5, f"Tích lũy 30d:\n{fmt_pct(pct_30, 1)}", ha="center", va="bottom", fontsize=10, color=GRAY_TEXT)

    ax.set_xlabel("Số ngày giữa 2 lần mua", fontsize=13, color=GRAY_TEXT)
    ax.set_xlim(0, 32.5)
    ax.set_ylim(0, max(shares) * 1.35)

    # Custom x ticks
    ax.set_xticks([1, 3, 7, 14, 21, 30, 31])
    ax.set_xticklabels(["1", "3", "7", "14", "21", "30", ">30"], fontsize=11)

    # Bỏ trục Y vì đã có nhãn
    hide_value_axis(ax, axis="y")
    ax.set_axisbelow(True)

    return fig


def fig_l2b(gaps: pd.Series) -> plt.Figure:
    """S5_ipt_cumulative: Cumulative percentage of repeat purchases up to 30 days (Linear scale).

    Storytelling update:
    - Đổi sang thang thường tuyến tính 1–30 ngày.
    - Vùng dưới đường từ 1 đến 7 ngày tô đỏ nhạt (BAND).
    - Đường nét đứt gióng về 2 trục CHỈ cho mốc 7 ngày.
    - Mốc 7 ngày tô đỏ, chữ lớn; 3 và 14 ngày tô xám, chữ nhỏ, đặt phía trên bên trái của chấm.
    - Trục x: 'Số ngày từ lần mua trước'; trục y: '% lần mua lại đã diễn ra'.
    - Bỏ grid dọc, chỉ giữ 2–3 grid ngang mảnh.
    """
    set_slide_style()
    fig, ax = plt.subplots(figsize=SLIDE_HALF)

    clean_gaps = ipt_gap_distribution(gaps)
    total_gaps = len(clean_gaps)

    # Linear scale 1 to 30 days
    days_range = np.arange(1, 31)
    cum_pcts = np.array([float((clean_gaps <= d).mean() * 100.0) for d in days_range])

    # 1. Shaded area under the curve from day 1 to day 7 in BAND (#FDE8EA)
    days_7 = np.arange(1, 8)
    cum_7 = cum_pcts[:7]
    ax.fill_between(days_7, 0, cum_7, color=BAND, alpha=0.9, zorder=1)

    # 2. Cumulative curve in INK
    ax.plot(days_range, cum_pcts, color=INK, lw=2.2, zorder=3)

    # Key points: 3, 7, 14 days
    pct_3 = float((clean_gaps <= 3).mean() * 100.0)
    pct_7 = float((clean_gaps <= 7).mean() * 100.0)
    pct_14 = float((clean_gaps <= 14).mean() * 100.0)

    # Dotted projection lines ONLY for 7 days
    ax.plot([0, 7], [pct_7, pct_7], color=RED, linestyle=":", lw=1.2, zorder=2)
    ax.plot([7, 7], [0, pct_7], color=RED, linestyle=":", lw=1.2, zorder=2)

    # Markers
    ax.scatter(3, pct_3, color=GRAY_TEXT, s=50, edgecolor="white", lw=1.0, zorder=4)
    ax.scatter(14, pct_14, color=GRAY_TEXT, s=50, edgecolor="white", lw=1.0, zorder=4)
    ax.scatter(7, pct_7, color=RED, s=120, edgecolor="white", lw=1.5, zorder=5)

    # Labels placed ABOVE-LEFT of the dots to prevent intersection with the curve
    ax.text(
        2.7,
        pct_3 + 3.0,
        f"3 ngày: {fmt_pct(pct_3, 1)}",
        ha="right",
        va="bottom",
        fontsize=11,
        color=GRAY_TEXT,
        fontweight="bold",
        zorder=5,
    )
    ax.text(
        6.7,
        pct_7 + 3.5,
        f"{fmt_pct(pct_7, 1)} trong ≤ 7 ngày",
        ha="right",
        va="bottom",
        fontsize=13,
        color=RED,
        fontweight="bold",
        zorder=5,
    )
    ax.text(
        13.7,
        pct_14 + 2.5,
        f"14 ngày: {fmt_pct(pct_14, 1)}",
        ha="right",
        va="bottom",
        fontsize=11,
        color=GRAY_TEXT,
        fontweight="bold",
        zorder=5,
    )

    # Axes styling: linear scale, x: 0 to 30, y: 0 to 105
    ax.set_xlabel("Số ngày từ lần mua trước", fontsize=13, color=GRAY_TEXT)
    ax.set_ylabel("% lần mua lại đã diễn ra", fontsize=13, color=GRAY_TEXT)
    ax.set_xlim(0, 31)
    ax.set_ylim(0, 105)
    pct_axis(ax, "y")

    # Grid: no vertical grid, only 2-3 thin horizontal grid lines
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color="#E5E7EB", linestyle="-", lw=0.8)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks([1, 7, 14, 21, 30])
    ax.set_axisbelow(True)

    return fig


def fig_l3(cutoff_rates_df: pd.DataFrame) -> plt.Figure:
    """S6_label_stability: Observed 7-day return rates across 81 cutoffs.

    Storytelling update:
    - Bỏ marker ở mọi điểm, chỉ giữ line trơn màu đỏ.
    - Bỏ dải tô nhạt p10–p90.
    - Bỏ hộp legend; ghi trực tiếp 'Median 53,1%' trên đường nét đứt xám.
    - Bỏ box chú thích thừa trên chart.
    """
    set_slide_style()
    fig, ax = plt.subplots(figsize=SLIDE_FULL)

    df = cutoff_rates_df[(cutoff_rates_df["cutoff_day"] >= 140) & (cutoff_rates_df["cutoff_day"] <= 700)].copy()
    med = float(df["return_rate"].median() * 100.0)

    # Smooth red line without markers
    ax.plot(df["cutoff_day"], df["return_rate"] * 100.0, color=RED, lw=2.2, zorder=3)

    # Median dashed line - label placed directly above the line at the right
    ax.axhline(med, color=GRAY_TEXT, linestyle="--", lw=1.5, zorder=2)
    ax.text(
        705,
        med + 0.6,
        f"Median {fmt_pct(med, 1)}",
        va="bottom",
        ha="left",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
    )

    ax.set_xlabel("Ngày quan sát (DAY cutoff)", fontsize=13, color=GRAY_TEXT)
    ax.set_ylabel("Tỷ lệ quay lại 7 ngày (%)", fontsize=13, color=GRAY_TEXT)
    ax.set_xlim(130, 750)
    ax.set_ylim(40, 65)
    pct_axis(ax, "y")
    ax.set_axisbelow(True)

    return fig


# ==============================================================================
# Slide Figures: Commercial Basket & Department (C1, C2)
# ==============================================================================

def fig_c1(bands_df: pd.DataFrame) -> plt.Figure:
    """S11_basket_bands: Quick-trip vs Stock-up basket size disparity.

    Storytelling update:
    - Xóa hộp legend.
    - Bỏ trục Y và grid trục Y (nhãn số trên đầu mỗi bar).
    - Ghi trực tiếp '% số lần mua' (xám) và '% doanh thu' (đỏ) ngay trên cặp cột đầu tiên.
    """
    set_slide_style()
    fig, ax = plt.subplots(figsize=SLIDE_HALF)

    x = np.arange(len(bands_df))
    width = 0.36

    pct_baskets = bands_df["pct_baskets"] * 100.0 if bands_df["pct_baskets"].max() <= 1.0 else bands_df["pct_baskets"]
    pct_spend = bands_df["pct_spend"] * 100.0 if bands_df["pct_spend"].max() <= 1.0 else bands_df["pct_spend"]

    bars1 = ax.bar(x - width / 2.0, pct_baskets, width=width, color=GRAY_BAR, edgecolor="white", lw=1.2)
    bars2 = ax.bar(x + width / 2.0, pct_spend, width=width, color=RED, edgecolor="white", lw=1.2)

    # Label on bars with decimal comma
    for b in bars1:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2.0, h + 0.8, fmt_pct(h, 1), ha="center", va="bottom", fontsize=11)

    for b in bars2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2.0, h + 0.8, fmt_pct(h, 1), ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Direct label on first pair of bars (thay thế hoàn toàn hộp legend)
    h1_first = pct_baskets.iloc[0]
    h2_first = pct_spend.iloc[0]
    ax.text(
        0 - width / 2.0,
        h1_first + 4.2,
        "% số lần mua",
        ha="center",
        va="bottom",
        fontsize=11,
        color=GRAY_TEXT,
        fontweight="bold",
    )
    ax.text(
        0 + width / 2.0,
        h2_first + 4.2,
        "% doanh thu",
        ha="center",
        va="bottom",
        fontsize=11,
        color=RED,
        fontweight="bold",
    )

    # X labels with median spend
    x_labels = []
    for _, row in bands_df.iterrows():
        b_name = str(row["band"])
        med_sp = float(row.get("median_spend", row.get("spend_p50", 0.0)))
        x_labels.append(f"{b_name} món\n(Median {fmt_money(med_sp, 1)})")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylim(0, 60)

    # Bỏ trục Y và grid trục Y
    hide_value_axis(ax, axis="y")
    ax.set_axisbelow(True)

    return fig


def fig_c2(dept_summary_df: pd.DataFrame) -> plt.Figure:
    """S11_department_pareto: Merchandise department Pareto spend concentration.

    Storytelling update:
    - 4 department đầu tô đỏ, tất cả còn lại (kể cả 'Khác') tô xám. Bỏ màu hồng nhạt.
    - Bỏ trục X và grid trục X.
    - Dời cột 'Tích lũy' sang phải (x ≈ 75%).
    """
    set_slide_style()
    fig, ax = plt.subplots(figsize=SLIDE_HALF)

    df = dept_summary_df.head(10).copy()
    dept_col = "department" if "department" in df.columns else "DEPARTMENT"

    other_spend = float(dept_summary_df.iloc[10:]["spend"].sum()) if len(dept_summary_df) > 10 else 0.0
    other_share = float(dept_summary_df.iloc[10:]["spend_share"].sum()) if len(dept_summary_df) > 10 else 0.0

    if other_spend > 0:
        other_row = pd.DataFrame([{
            dept_col: "Khác",
            "spend": other_spend,
            "spend_share": other_share,
            "cum_spend_share": 1.0,
        }])
        df = pd.concat([df, other_row], ignore_index=True)

    # Sort descending so top department is at top
    df = df.iloc[::-1].reset_index(drop=True)
    n = len(df)
    y_pos = np.arange(n)

    shares = df["spend_share"] * 100.0 if df["spend_share"].max() <= 1.0 else df["spend_share"]

    # Monochromatic: Top 4 RED, all others GRAY_BAR
    colors = []
    for dept in df[dept_col]:
        if str(dept).upper() in ["GROCERY", "DRUG GM", "PRODUCE", "MEAT"]:
            colors.append(RED)
        else:
            colors.append(GRAY_BAR)

    ax.barh(y_pos, shares, color=colors, edgecolor="white", lw=1.2, height=0.68)

    # Text at bar end and cumulative text shifted to x = 75%
    for i, row in df.iterrows():
        sh = shares.iloc[i]
        cum = float(row["cum_spend_share"]) * 100.0
        ax.text(sh + 0.8, i, fmt_pct(sh, 1), va="center", ha="left", fontsize=11, fontweight="bold")
        if str(row[dept_col]).lower() != "khác":
            ax.text(75, i, f"Tích lũy: {fmt_pct(cum, 1)}", va="center", ha="left", fontsize=10.5, color=GRAY_TEXT)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df[dept_col], fontsize=11)
    ax.set_xlim(0, 96)

    # Bỏ trục X và grid trục X
    hide_value_axis(ax, axis="x")

    # Footnote below chart
    fig.text(
        0.12,
        0.01,
        "5 commodity thiết yếu: Soft drinks, Milk, Bread, Bag snacks, Beef",
        fontsize=10.5,
        fontstyle="italic",
        color=GRAY_TEXT,
    )

    ax.set_axisbelow(True)
    return fig


# ==============================================================================
# Slide Figures: Promotion, Campaign & Demographic (P1, P2, D1, Q1)
# ==============================================================================

def fig_p1(weekly_promo_df: pd.DataFrame) -> plt.Figure:
    """S12_weekly_promo: Weekly promotional spend share stabilizing at 46.8%."""
    set_slide_style()
    fig, ax = plt.subplots(figsize=SLIDE_HALF)

    # Exclude partial weeks (week 1 and last week)
    df = weekly_promo_df.copy()
    if len(df) > 2:
        df = df[(df["WEEK_NO"] > 1) & (df["WEEK_NO"] < df["WEEK_NO"].max())].reset_index(drop=True)

    shares = df["promo_spend_share"] * 100.0 if df["promo_spend_share"].max() <= 1.0 else df["promo_spend_share"]
    mean_val = float(shares.mean())

    ax.plot(df["WEEK_NO"], shares, color=RED, lw=2.2, zorder=3)

    # Mean dashed line with label ABOVE the line to prevent intersection
    ax.axhline(mean_val, color=INK, linestyle="--", lw=1.2, zorder=2)
    ax.text(
        df["WEEK_NO"].iloc[-1] - 1,
        mean_val + 1.8,
        f"Trung bình {fmt_pct(mean_val, 1)}",
        ha="right",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )

    ax.set_xlabel("Tuần lịch (WEEK_NO)", fontsize=13, color=GRAY_TEXT)
    ax.set_ylabel("Tỷ trọng doanh thu giảm giá (%)", fontsize=13, color=GRAY_TEXT)
    ax.set_ylim(0, 70)
    pct_axis(ax, "y")
    ax.set_axisbelow(True)

    return fig


def fig_p2(reach_by_f: pd.DataFrame | dict[str, pd.DataFrame]) -> plt.Figure:
    """S12_campaign_reach: Extreme selection bias in campaign reach across frequency bins.

    Storytelling update:
    - Cột F cao nhất (> 1,52) tô đỏ, 4 cột còn lại tô xám.
    - Bỏ trục Y và grid trục Y.
    """
    set_slide_style()
    fig, ax = plt.subplots(figsize=SLIDE_HALF)

    if isinstance(reach_by_f, dict):
        df = reach_by_f["by_f"].copy()
    else:
        df = reach_by_f.copy()

    f_col = [c for c in df.columns if "f" in c.lower()][0]
    df = df[df[f_col].astype(str) != "Lịch sử ngắn (<28d)"].copy()

    f_map = {
        "≤ 0.30": "≤ 0,30",
        "0.30–0.57": "0,30–0,57",
        "0.57–0.96": "0,57–0,96",
        "0.96–1.52": "0,96–1,52",
        "> 1.52": "> 1,52",
    }
    x_labels = [f_map.get(str(val), str(val)) for val in df[f_col]]

    x = np.arange(len(df))
    reach_pct = df["reach_rate"] * 100.0 if df["reach_rate"].max() <= 1.0 else df["reach_rate"]

    # Monochromatic: highest bar RED, rest GRAY_BAR
    colors = [RED if i == len(df) - 1 else GRAY_BAR for i in range(len(df))]
    bars = ax.bar(x, reach_pct, color=colors, width=0.55, edgecolor="white", lw=1.2)

    # Labels on bars with decimal comma
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2.0, h + 0.8, fmt_pct(h, 1), ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel("Tần suất mua sắm (F)", fontsize=13, color=GRAY_TEXT)
    ax.set_ylim(0, 48)

    # Bỏ trục Y và grid trục Y
    hide_value_axis(ax, axis="y")

    # Annotate highest bar
    annotate_point(
        ax,
        xy=(x[-1], reach_pct.iloc[-1]),
        text="Gấp khoảng 170 lần nhóm F thấp nhất",
        xytext=(x[-1] - 0.8, 43),
        fontsize=12,
        color=RED,
    )

    ax.set_axisbelow(True)
    return fig


def fig_d1(cov_df: pd.DataFrame) -> plt.Figure:
    """S2_table_coverage: Household coverage comparison across raw transaction and secondary tables.

    Storytelling update:
    - Xám toàn bộ 3 bảng kia, chỉ 'Nhân khẩu học' tô đỏ (lưu ý chính về độ phủ chỉ 32%).
    - Bỏ trục X và grid trục X.
    """
    set_slide_style()
    fig, ax = plt.subplots(figsize=SLIDE_HALF)

    df = cov_df.sort_values("pct_households", ascending=True).reset_index(drop=True)
    n = len(df)
    y_pos = np.arange(n)

    # Colors: Demographic RED, others GRAY_BAR
    colors = [RED if "nhân khẩu" in str(name).lower() or "demographic" in str(name).lower() else GRAY_BAR for name in df["table_name"]]

    ax.barh(y_pos, df["pct_households"], color=colors, edgecolor="white", lw=1.5, height=0.55)

    # Text at end of bar: xx,x% (n hộ) with Vietnamese format
    for i, row in df.iterrows():
        pct = row["pct_households"]
        n_hh = int(row["n_households"])
        text_color = RED if colors[i] == RED else INK
        ax.text(
            pct + 1.2,
            i,
            f"{fmt_pct(pct, 1)} ({fmt_num(n_hh)} hộ)",
            va="center",
            ha="left",
            fontsize=11.5,
            fontweight="bold",
            color=text_color,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["table_name"], fontsize=12)
    ax.set_xlim(0, 125)

    # Bỏ trục X và grid trục X
    hide_value_axis(ax, axis="x")
    ax.set_axisbelow(True)
    return fig


def fig_q1(demo_cov: pd.DataFrame) -> plt.Figure:
    """S13_demographic_bias: Selection bias comparing F distribution of demographic vs non-demographic households.

    Storytelling update:
    - Xóa hộp legend.
    - Bỏ trục Y và grid trục Y.
    - Ghi trực tiếp 'Có thông tin khảo sát' (đỏ) và 'Không có' (xám) trên cặp cột đầu tiên.
    """
    set_slide_style()
    fig, ax = plt.subplots(figsize=SLIDE_HALF)

    f_bins_raw = ["≤ 0.30", "0.30–0.57", "0.57–0.96", "0.96–1.52", "> 1.52"]
    f_bins_vn = ["≤ 0,30", "0,30–0,57", "0,57–0,96", "0,96–1,52", "> 1,52"]
    x = np.arange(len(f_bins_vn))
    width = 0.36

    row_demo = demo_cov[demo_cov["has_demographic"] == 1].iloc[0]
    row_no_demo = demo_cov[demo_cov["has_demographic"] == 0].iloc[0]

    demo_pcts = [float(row_demo.get(f"f_{b}", row_demo.get(f"f_{b_vn}", 0.0))) * 100.0 for b, b_vn in zip(f_bins_raw, f_bins_vn)]
    no_demo_pcts = [float(row_no_demo.get(f"f_{b}", row_no_demo.get(f"f_{b_vn}", 0.0))) * 100.0 for b, b_vn in zip(f_bins_raw, f_bins_vn)]

    bars1 = ax.bar(x - width / 2.0, demo_pcts, width=width, color=RED, edgecolor="white", lw=1.2)
    bars2 = ax.bar(x + width / 2.0, no_demo_pcts, width=width, color=GRAY_BAR, edgecolor="white", lw=1.2)

    # Text on bars with decimal comma
    for b in bars1:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2.0, h + 0.8, fmt_pct(h, 1), ha="center", va="bottom", fontsize=10.5, fontweight="bold")

    for b in bars2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2.0, h + 0.8, fmt_pct(h, 1), ha="center", va="bottom", fontsize=10.5)

    # Direct labels on first pair of bars (thay thế hoàn toàn hộp legend)
    h1_first = demo_pcts[0]
    h2_first = no_demo_pcts[0]
    ax.text(
        0 - width / 2.0,
        h1_first + 3.8,
        "Có thông tin\nkhảo sát",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color=RED,
        fontweight="bold",
    )
    ax.text(
        0 + width / 2.0,
        h2_first + 3.8,
        "Không có",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color=GRAY_TEXT,
        fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(f_bins_vn, fontsize=11)
    ax.set_xlabel("Tần suất mua sắm (F)", fontsize=13, color=GRAY_TEXT)
    ax.set_ylim(0, 50)

    # Bỏ trục Y và grid trục Y
    hide_value_axis(ax, axis="y")
    ax.set_axisbelow(True)

    return fig

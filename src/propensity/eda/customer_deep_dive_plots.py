"""Customer deep-dive visualizations: Q-F continuity, Q-CV diagnostics, Q-M spend decomposition, and Q-IPT profile."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from propensity.eda.customer_deep_dive import CustomerDeepDiveReport

# Visual style palette
_COLOR_PRIMARY = "#1E3D59"       # Deep slate blue
_COLOR_SECONDARY = "#17B978"     # Emerald teal
_COLOR_ACCENT = "#FF6E40"        # Coral orange
_COLOR_WARN = "#F6C90E"          # Amber gold
_COLOR_MUTED = "#8D99AE"         # Muted gray
_COLOR_BG = "#F8F9FA"            # Light background


def plot_f_continuity(
    report: CustomerDeepDiveReport,
    top_n: int = 30,
) -> Figure:
    """Plot Q-F continuity diagnostics: scatter F x weekly_coverage and high-F activity timeline.

    Parameters
    ----------
    report : CustomerDeepDiveReport
        Output of build_customer_deep_dive_report.
    top_n : int
        Number of top high-F households to display in the weekly activity heatmap.

    Returns
    -------
    Figure
        Matplotlib Figure with 2 panels.
    """
    cont = report.continuity
    high_f_cls = report.high_f_classified
    pdays = report.purchase_days
    thr = report.thresholds

    fig, (ax_scatter, ax_heat) = plt.subplots(
        1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.1, 1.2]}
    )
    fig.patch.set_facecolor("#FFFFFF")

    # -------------------------------------------------------------
    # Panel 1: Scatter F x weekly_coverage colored by span_days
    # -------------------------------------------------------------
    q_f = float(cont["F"].quantile(thr.high_f_quantile))
    sc = ax_scatter.scatter(
        cont["F"],
        cont["weekly_coverage"] * 100.0,
        c=cont["span_days"],
        cmap="viridis",
        alpha=0.6,
        s=25,
        edgecolor="none",
    )
    cbar = fig.colorbar(sc, ax=ax_scatter, pad=0.02)
    cbar.set_label("Độ dài lịch sử hoạt động (span_days)", fontsize=10)

    # Threshold guidelines
    ax_scatter.axvline(q_f, color=_COLOR_ACCENT, linestyle="--", linewidth=1.5, alpha=0.9)
    ax_scatter.axhline(thr.continuous_coverage * 100.0, color=_COLOR_SECONDARY, linestyle="--", linewidth=1.5, alpha=0.9)

    ax_scatter.text(
        q_f + 0.05, 15,
        f"Ngưỡng F top 20%:\nF ≥ {q_f:.2f} ngày/tuần",
        fontsize=9, color=_COLOR_ACCENT, fontweight="bold",
    )
    ax_scatter.text(
        0.1, thr.continuous_coverage * 100.0 + 2,
        f"Độ phủ liên tục ≥ {thr.continuous_coverage*100:.0f}% số tuần",
        fontsize=9, color=_COLOR_SECONDARY, fontweight="bold",
    )

    # Highlight High-F continuous count
    n_cont = int(report.high_f_summary.loc[report.high_f_summary["f_pattern"] == "liên tục", "n_households"].values[0])
    pct_cont = float(report.high_f_summary.loc[report.high_f_summary["f_pattern"] == "liên tục", "pct_households"].values[0])
    ax_scatter.annotate(
        f"Nhóm Mua Liên Tục:\n{n_cont:,} hộ ({pct_cont:.1f}% của Top 20% F)\nĐộ phủ cao suốt toàn bộ span",
        xy=(max(q_f + 0.5, cont['F'].quantile(0.90)), 90),
        xytext=(q_f + 1.2, 50),
        arrowprops=dict(facecolor=_COLOR_PRIMARY, arrowstyle="->", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF4F6", edgecolor=_COLOR_PRIMARY, alpha=0.9),
        fontsize=9, fontweight="semibold",
    )

    ax_scatter.set_title("A. Tần suất (F) vs Độ Phủ Tuần (Weekly Coverage)", fontsize=12, fontweight="bold", pad=12)
    ax_scatter.set_xlabel("F: Số ngày mua bình quân mỗi tuần", fontsize=10)
    ax_scatter.set_ylabel("Độ phủ tuần: % số tuần có phát sinh giao dịch", fontsize=10)
    ax_scatter.set_ylim(-2, 105)
    ax_scatter.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # Panel 2: Weekly Activity Timeline Heatmap for Top N High-F
    # -------------------------------------------------------------
    # Select top N high-F households sorted by weekly_coverage
    top_hh = high_f_cls.sort_values("weekly_coverage", ascending=True).tail(top_n)
    hh_keys = top_hh["household_key"].values

    # Determine max relative week
    first_day_map = report.continuity.set_index("household_key")["first_day"].to_dict()
    sub_pdays = pdays[pdays["household_key"].isin(hh_keys)].copy()
    sub_pdays["first_day"] = sub_pdays["household_key"].map(first_day_map)
    sub_pdays["rel_week"] = (sub_pdays["DAY"] - sub_pdays["first_day"]) // 7

    max_rel_week = int(sub_pdays["rel_week"].max()) if not sub_pdays.empty else 100
    max_cols = min(max_rel_week + 1, 104)

    # Build binary matrix: [top_n, max_cols]
    matrix = np.zeros((len(hh_keys), max_cols), dtype=float)
    for row_idx, hhk in enumerate(hh_keys):
        hh_weeks = sub_pdays.loc[sub_pdays["household_key"] == hhk, "rel_week"].values
        valid_w = hh_weeks[(hh_weeks >= 0) & (hh_weeks < max_cols)]
        matrix[row_idx, valid_w] = 1.0

    ax_heat.imshow(matrix, aspect="auto", cmap="Blues", interpolation="nearest", origin="lower")

    y_labels = [f"HH {hhk} (cov: {cov*100:.0f}%)" for hhk, cov in zip(hh_keys, top_hh["weekly_coverage"])]
    ax_heat.set_yticks(np.arange(len(hh_keys)))
    ax_heat.set_yticklabels(y_labels, fontsize=8)
    ax_heat.set_title(f"B. Dòng Hoạt Động Theo Tuần (Top {top_n} Hộ F Cao Nhất)", fontsize=12, fontweight="bold", pad=12)
    ax_heat.set_xlabel("Tuần hoạt động tương đối (tính từ ngày mua đầu tiên của hộ)", fontsize=10)
    ax_heat.set_ylabel("Hộ gia đình (xếp theo độ phủ tăng dần)", fontsize=10)

    fig.suptitle(
        "Q-F: ĐÁNH GIÁ TÍNH LIÊN TỤC CỦA TẦN SUẤT MUA SẮM (FREQUENCY CONTINUITY)\n"
        "92% hộ F cao duy trì nhịp mua liên tục qua các tuần — F là tín hiệu bền vững, không phải mua dồn cục bộ",
        fontsize=13, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def plot_cv_diagnostics(report: CustomerDeepDiveReport) -> Figure:
    """Plot Q-CV diagnostics: CV vs sample size, driver breakdown, and max-gap impact.

    Parameters
    ----------
    report : CustomerDeepDiveReport
        Output of build_customer_deep_dive_report.

    Returns
    -------
    Figure
        Matplotlib Figure with 3 panels.
    """
    diag = report.cv_diagnostics
    cv_sum = report.cv_driver_summary
    thr = report.thresholds

    fig, (ax_scatter, ax_bar, ax_hist) = plt.subplots(
        1, 3, figsize=(18, 6.2), gridspec_kw={"width_ratios": [1.1, 0.9, 1.0]}
    )
    fig.patch.set_facecolor("#FFFFFF")

    # -------------------------------------------------------------
    # Panel A: Scatter CV x n_gaps (log scale on X)
    # -------------------------------------------------------------
    clean_diag = diag.dropna(subset=["cv_ipt", "n_gaps"]).copy()
    color_map = {
        "đều đặn": _COLOR_SECONDARY,
        "thất thường thật": _COLOR_ACCENT,
        "do nghỉ dài": _COLOR_WARN,
        "ít lần mua": _COLOR_MUTED,
        "không đủ dữ liệu": "#CCCCCC",
    }
    point_colors = [color_map.get(d, "#999999") for d in clean_diag["cv_driver"]]

    ax_scatter.scatter(
        clean_diag["n_gaps"],
        clean_diag["cv_ipt"],
        c=point_colors,
        alpha=0.5,
        s=18,
        edgecolor="none",
    )
    ax_scatter.set_xscale("log")
    ax_scatter.axhline(thr.cv_high, color=_COLOR_PRIMARY, linestyle="--", linewidth=1.2, label=f"Ngưỡng CV = {thr.cv_high:.1f}")
    ax_scatter.axvline(thr.min_gaps, color=_COLOR_MUTED, linestyle=":", linewidth=1.2, label=f"Ngưỡng n_gaps = {thr.min_gaps}")

    ax_scatter.set_title("A. CV vs Số Lần Mua (n_gaps, Log X)", fontsize=11, fontweight="bold", pad=10)
    ax_scatter.set_xlabel("Số khoảng cách mua (n_gaps, thang log)", fontsize=9)
    ax_scatter.set_ylabel("Hệ số biến thiên IPT (cv_ipt)", fontsize=9)
    ax_scatter.set_ylim(-0.05, 3.5)
    ax_scatter.grid(True, linestyle=":", alpha=0.6)
    ax_scatter.legend(loc="upper right", fontsize=8)

    # -------------------------------------------------------------
    # Panel B: Bar chart of cv_driver for CV > 1.0
    # -------------------------------------------------------------
    drivers = cv_sum["cv_driver"].tolist()
    counts = cv_sum["n_households"].tolist()
    pcts = cv_sum["pct_share"].tolist()
    bar_colors = [_COLOR_ACCENT, _COLOR_WARN, _COLOR_MUTED]

    bars = ax_bar.bar(drivers, counts, color=bar_colors, width=0.55, edgecolor="none")
    for bar, pct, cnt in zip(bars, pcts, counts):
        h = bar.get_height()
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2, h + 15,
            f"{cnt:,} hộ\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax_bar.set_title("B. Nguyên Nhân Gây CV > 1.0 (N = 1,267 hộ)", fontsize=11, fontweight="bold", pad=10)
    ax_bar.set_ylabel("Số lượng hộ gia đình", fontsize=9)
    ax_bar.set_ylim(0, max(counts) * 1.25)
    ax_bar.grid(axis="y", linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # Panel C: Histogram of CV before vs after dropping max gap
    # -------------------------------------------------------------
    valid_cv = clean_diag[clean_diag["cv_ipt"] > thr.cv_high].dropna(subset=["cv_drop_max"])
    bins = np.linspace(0.2, 2.5, 35)

    ax_hist.hist(
        valid_cv["cv_ipt"], bins=bins, alpha=0.5, color=_COLOR_ACCENT,
        label=f"CV ban đầu (Median: {valid_cv['cv_ipt'].median():.2f})", edgecolor="white"
    )
    ax_hist.hist(
        valid_cv["cv_drop_max"], bins=bins, alpha=0.6, color=_COLOR_SECONDARY,
        label=f"CV sau khi bỏ max gap (Median: {valid_cv['cv_drop_max'].median():.2f})", edgecolor="white"
    )
    ax_hist.axvline(thr.cv_high, color=_COLOR_PRIMARY, linestyle="--", linewidth=1.5, label="Ngưỡng CV = 1.0")

    ax_hist.set_title("C. Tác Động Khi Loại Bỏ Kỳ Nghỉ Dài Nhất", fontsize=11, fontweight="bold", pad=10)
    ax_hist.set_xlabel("Hệ số biến thiên (CV)", fontsize=9)
    ax_hist.set_ylabel("Tần số hộ gia đình", fontsize=9)
    ax_hist.legend(loc="upper right", fontsize=8)
    ax_hist.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        "Q-CV: GIẢI PHẪU NGUYÊN NHÂN BIẾN THIÊN NHỊP MUA (CV DIAGNOSTICS)\n"
        "35.6% hộ có CV cao thực chất mua rất đều đặn, chỉ bị lệch do 1-2 kỳ nghỉ dài (vacation bias)",
        fontsize=13, fontweight="bold", y=0.98
    )
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.06, right=0.96, wspace=0.28)
    return fig


def plot_spend_decomposition(report: CustomerDeepDiveReport) -> Figure:
    """Plot Q-M spend variance decomposition and hexbin of log(F) vs log(spend/trip).

    Parameters
    ----------
    report : CustomerDeepDiveReport
        Output of build_customer_deep_dive_report.

    Returns
    -------
    Figure
        Matplotlib Figure with 2 panels.
    """
    decomp = report.spend_decomposition
    df_shares = decomp.variance_shares

    fig, (ax_bar, ax_hex) = plt.subplots(
        1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.1, 1.0]}
    )
    fig.patch.set_facecolor("#FFFFFF")

    # -------------------------------------------------------------
    # Panel A: Horizontal Bar of Variance Shares
    # -------------------------------------------------------------
    components = df_shares["component"].tolist()[::-1]
    pcts = df_shares["variance_share_pct"].tolist()[::-1]
    corrs = df_shares["spearman_corr_with_spend"].tolist()[::-1]
    bar_colors = [_COLOR_MUTED, _COLOR_SECONDARY, _COLOR_PRIMARY]

    y_pos = np.arange(len(components))
    bars = ax_bar.barh(y_pos, pcts, color=bar_colors, height=0.45, edgecolor="none")

    for bar, pct, corr in zip(bars, pcts, corrs):
        w = bar.get_width()
        ax_bar.text(
            w + 1.2, bar.get_y() + bar.get_height() / 2,
            f"{pct:.2f}% (Tương quan Spearman: {corr:+.3f})",
            va="center", fontsize=9, fontweight="bold",
        )

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(components, fontsize=9, fontweight="semibold")
    ax_bar.set_xlabel("% Đóng góp vào phương sai ln(spend/tuần)", fontsize=10)
    ax_bar.set_xlim(0, 85)
    ax_bar.set_title("A. Phân Rã Phương Sai Biến Thiên Doanh Thu Hàng Tuần", fontsize=11, fontweight="bold", pad=12)
    ax_bar.grid(axis="x", linestyle=":", alpha=0.6)

    # Decision rule text box
    ax_bar.text(
        0.04, 0.12,
        f"QUY TẮC QUYẾT ĐỊNH (DECISION RULE):\n"
        f"• Đóng góp của F đạt {decomp.f_share*100:.1f}% (≥ {report.thresholds.f_share_redundant*100:.0f}%)\n"
        f"• Biến spend_per_week thừa thông tin (>69% bị chi phối bởi tần suất F)\n"
        f"• Kết luận: Loại bỏ spend/tuần, chỉ giữ lại F và spend/trip",
        transform=ax_bar.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3E0", edgecolor=_COLOR_ACCENT, lw=1.2),
    )

    # -------------------------------------------------------------
    # Panel B: Hexbin of log(F) vs log(spend/trip)
    # -------------------------------------------------------------
    pdays = report.purchase_days
    summary = pdays.groupby("household_key").agg(
        n_days=("DAY", "count"),
        first_d=("DAY", "min"),
        last_d=("DAY", "max"),
        n_trips=("n_trips_day", "sum"),
        total_sp=("spend_day", "sum"),
    ).reset_index()
    summary["span_days"] = summary["last_d"] - summary["first_d"] + 1
    summary = summary[summary["span_days"] >= 28].copy()
    summary["F"] = summary["n_days"] / (summary["span_days"] / 7.0)
    summary["spend_per_trip"] = summary["total_sp"] / summary["n_trips"]

    clean = summary[(summary["F"] > 0) & (summary["spend_per_trip"] > 0)]
    log_f = np.log(clean["F"])
    log_spt = np.log(clean["spend_per_trip"])

    hb = ax_hex.hexbin(log_f, log_spt, gridsize=32, cmap="YlGnBu", mincnt=1)
    cb = fig.colorbar(hb, ax=ax_hex, pad=0.02)
    cb.set_label("Mật độ hộ gia đình", fontsize=9)

    # Compute correlation between log F and log spend_per_trip
    corr_axes = float(clean["F"].rank().corr(clean["spend_per_trip"].rank()))

    ax_hex.set_title("B. Mối Quan Hệ Giữa Tần Suất (F) và Chi Tiêu/Trip", fontsize=11, fontweight="bold", pad=12)
    ax_hex.set_xlabel("ln(F): Logarit số ngày mua/tuần", fontsize=10)
    ax_hex.set_ylabel("ln(spend_per_trip): Logarit chi tiêu/chuyến", fontsize=10)
    ax_hex.grid(True, linestyle=":", alpha=0.5)

    ax_hex.text(
        0.05, 0.90,
        f"Tương quan Spearman giữa F và Spend/trip: {corr_axes:+.3f}\n"
        "(Gần như trực giao, thể hiện 2 khía cạnh hành vi độc lập)",
        transform=ax_hex.transAxes,
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9", edgecolor=_COLOR_SECONDARY, lw=1.1),
    )

    fig.suptitle(
        "Q-M: PHÂN RÃ PHƯƠNG SAI SPEND/TUẦN (SPEND DECOMPOSITION)\n"
        "F chi phối gần 70% biến thiên doanh thu — spend/tuần bị trùng lặp thông tin, nên giữ cặp tách bạch: (F, spend/trip)",
        fontsize=13, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def plot_ipt_band_profile(report: CustomerDeepDiveReport) -> Figure:
    """Plot Q-IPT profile: key metrics across IPT bands and subtype breakdown of > 14 days.

    Parameters
    ----------
    report : CustomerDeepDiveReport
        Output of build_customer_deep_dive_report.

    Returns
    -------
    Figure
        Matplotlib Figure with 2 panels.
    """
    bands_df = report.ipt_band_profile
    subtypes_df = report.ipt_gt14_subtypes

    fig, (ax_bands, ax_sub) = plt.subplots(
        1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.1, 1.0]}
    )
    fig.patch.set_facecolor("#FFFFFF")

    # -------------------------------------------------------------
    # Panel A: Metrics across IPT Bands
    # -------------------------------------------------------------
    bands = bands_df["ipt_band"].tolist()
    x = np.arange(len(bands))
    width = 0.35

    pct_hh = bands_df["pct_households"].tolist()
    pct_spend = bands_df["pct_total_spend"].tolist()

    rects1 = ax_bands.bar(x - width/2, pct_hh, width, label="% Số Hộ Gia Đình", color=_COLOR_PRIMARY, edgecolor="none")
    rects2 = ax_bands.bar(x + width/2, pct_spend, width, label="% Đóng Góp Doanh Thu", color=_COLOR_SECONDARY, edgecolor="none")

    for r in rects1:
        h = r.get_height()
        ax_bands.text(r.get_x() + r.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for r in rects2:
        h = r.get_height()
        ax_bands.text(r.get_x() + r.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax_bands.set_xticks(x)
    ax_bands.set_xticklabels([f"Band {b}\n(IPT median)" for b in bands], fontsize=9)
    ax_bands.set_title("A. Tỷ Trọng Số Hộ và Doanh Thu Theo Nhóm IPT Median", fontsize=11, fontweight="bold", pad=12)
    ax_bands.set_ylabel("% Tổng toàn hệ thống", fontsize=10)
    ax_bands.set_ylim(0, max(max(pct_hh), max(pct_spend)) * 1.20)
    ax_bands.legend(loc="upper right", fontsize=9)
    ax_bands.grid(axis="y", linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # Panel B: Subtypes of Band > 14 days
    # -------------------------------------------------------------
    st_names = subtypes_df["ipt_subtype"].tolist()[::-1]
    st_counts = subtypes_df["n_households"].tolist()[::-1]
    st_pcts = subtypes_df["pct_subtypes"].tolist()[::-1]
    st_spt = subtypes_df["median_spend_per_trip"].tolist()[::-1]

    colors_subtype = [_COLOR_MUTED, _COLOR_SECONDARY, _COLOR_WARN, _COLOR_ACCENT, _COLOR_PRIMARY]

    y_pos = np.arange(len(st_names))
    bars = ax_sub.barh(y_pos, st_counts, color=colors_subtype, height=0.5, edgecolor="none")

    for bar, cnt, pct, spt in zip(bars, st_counts, st_pcts, st_spt):
        w = bar.get_width()
        spt_text = f"Spend/trip: ${spt:.1f}" if pd.notna(spt) else ""
        ax_sub.text(
            w + 2, bar.get_y() + bar.get_height() / 2,
            f"{cnt} hộ ({pct:.1f}%) | {spt_text}",
            va="center", fontsize=8.5, fontweight="bold",
        )

    ax_sub.set_yticks(y_pos)
    ax_sub.set_yticklabels(st_names, fontsize=9, fontweight="semibold")
    ax_sub.set_xlabel("Số lượng hộ gia đình", fontsize=10)
    ax_sub.set_xlim(0, max(st_counts) * 1.55)
    ax_sub.set_title("B. Phân Loại Bản Chất Nhóm Nhịp Mua Dài (> 14 ngày, N = 385)", fontsize=11, fontweight="bold", pad=12)
    ax_sub.grid(axis="x", linestyle=":", alpha=0.6)

    fig.suptitle(
        "Q-IPT: HÀNH VI KHÁCH HÀNG CÓ NHỊP MUA DÀI (IPT > 14 NGÀY)\n"
        "Nhóm >14 ngày chiếm 15.4% số hộ nhưng chỉ đóng góp 4.9% doanh thu; gồm cả hộ mua bổ sung, mua dự trữ và rời bỏ",
        fontsize=13, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig

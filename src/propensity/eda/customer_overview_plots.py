"""Visualizations for customer overview: distributions, cadence, and Lorenz curves."""

from __future__ import annotations

from typing import Final
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from propensity.eda.customer_overview import gini, lorenz_curve

# Color Palette
_COLOR_PRIMARY: Final = "#1E40AF"
_COLOR_ACCENT: Final = "#DC2626"
_COLOR_MEDIAN: Final = "#0F172A"
_COLOR_PERCENTILE: Final = "#64748B"
_COLOR_BAR: Final = "#93C5FD"
_COLOR_REF: Final = "#F59E0B"


def plot_customer_overview_histograms(
    summary: pd.DataFrame,
    exclude_short_span: bool = False,
) -> Figure:
    """Plot 5-panel distribution grid for customer F, M, and IPT cadence metrics.

    Parameters
    ----------
    summary : pd.DataFrame
        Output of build_household_summary.
    exclude_short_span : bool, default False
        Whether to filter out households with span_days < 28.

    Returns
    -------
    Figure
        Matplotlib figure containing the 5-panel diagnostic grid.
    """
    total_hh = len(summary)
    data = summary[~summary["short_span"]].copy() if exclude_short_span else summary.copy()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), dpi=150)
    axes_flat = axes.flat

    # -------------------------------------------------------------
    # Panel 1: Frequency F (purchase days per week)
    # -------------------------------------------------------------
    ax1 = axes_flat[0]
    s1 = data["F"].dropna()
    n_excluded1 = total_hh - len(s1)
    
    # Clip extreme tail for clean histogram display
    max_f = min(7.0, np.percentile(s1, 99.5) * 1.1)
    bins1 = np.linspace(0, max_f, 35)
    ax1.hist(s1, bins=bins1, color=_COLOR_BAR, edgecolor="white", alpha=0.85)
    
    med1 = s1.median()
    p10_1, p90_1 = np.percentile(s1, [10, 90])
    ax1.axvline(med1, color=_COLOR_MEDIAN, linestyle="-", linewidth=1.8, label=f"Median: {med1:.2f}")
    ax1.axvline(p10_1, color=_COLOR_PERCENTILE, linestyle="--", linewidth=1.3, label=f"p10: {p10_1:.2f}")
    ax1.axvline(p90_1, color=_COLOR_PERCENTILE, linestyle=":", linewidth=1.3, label=f"p90: {p90_1:.2f}")
    
    ax1.set_title("Tần suất mua hàng (F: Ngày mua / Tuần)", fontsize=11, fontweight="bold", color="#1F2937")
    ax1.set_xlabel("Số ngày mua mỗi tuần (F)", fontsize=9.5)
    ax1.set_ylabel("Số household", fontsize=9.5)
    ax1.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="#F8FAFC")
    ax1.text(0.03, 0.92, f"Loại bỏ (null/short): {n_excluded1}", transform=ax1.transAxes,
             fontsize=8.5, color="#64748B", bbox=dict(boxstyle="round,pad=0.2", facecolor="#F1F5F9", edgecolor="none"))
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel 2: Spend per week (Log Scale)
    # -------------------------------------------------------------
    ax2 = axes_flat[1]
    s2 = data["spend_per_week"].dropna()
    s2 = s2[s2 > 0]
    n_excluded2 = total_hh - len(s2)
    
    min_log2 = np.floor(np.log10(s2.min()))
    max_log2 = np.ceil(np.log10(s2.max()))
    bins2 = np.logspace(min_log2, max_log2, 35)
    
    ax2.hist(s2, bins=bins2, color=_COLOR_BAR, edgecolor="white", alpha=0.85)
    ax2.set_xscale("log")
    
    med2 = s2.median()
    p10_2, p90_2 = np.percentile(s2, [10, 90])
    ax2.axvline(med2, color=_COLOR_MEDIAN, linestyle="-", linewidth=1.8, label=f"Median: ${med2:.1f}")
    ax2.axvline(p10_2, color=_COLOR_PERCENTILE, linestyle="--", linewidth=1.3, label=f"p10: ${p10_2:.1f}")
    ax2.axvline(p90_2, color=_COLOR_PERCENTILE, linestyle=":", linewidth=1.3, label=f"p90: ${p90_2:.1f}")
    
    ax2.set_title("Chi tiêu hàng tuần (Spend / Tuần - Log scale)", fontsize=11, fontweight="bold", color="#1F2937")
    ax2.set_xlabel("Spend per week ($)", fontsize=9.5)
    ax2.set_ylabel("Số household", fontsize=9.5)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"${int(y):,}"))
    ax2.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="#F8FAFC")
    ax2.text(0.03, 0.92, f"Loại bỏ (null/short): {n_excluded2}", transform=ax2.transAxes,
             fontsize=8.5, color="#64748B", bbox=dict(boxstyle="round,pad=0.2", facecolor="#F1F5F9", edgecolor="none"))
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel 3: Spend per trip (Log Scale)
    # -------------------------------------------------------------
    ax3 = axes_flat[2]
    s3 = data["spend_per_trip"].dropna()
    s3 = s3[s3 > 0]
    n_excluded3 = total_hh - len(s3)
    
    min_log3 = np.floor(np.log10(s3.min()))
    max_log3 = np.ceil(np.log10(s3.max()))
    bins3 = np.logspace(min_log3, max_log3, 35)
    
    ax3.hist(s3, bins=bins3, color=_COLOR_BAR, edgecolor="white", alpha=0.85)
    ax3.set_xscale("log")
    
    med3 = s3.median()
    p10_3, p90_3 = np.percentile(s3, [10, 90])
    ax3.axvline(med3, color=_COLOR_MEDIAN, linestyle="-", linewidth=1.8, label=f"Median: ${med3:.1f}")
    ax3.axvline(p10_3, color=_COLOR_PERCENTILE, linestyle="--", linewidth=1.3, label=f"p10: ${p10_3:.1f}")
    ax3.axvline(p90_3, color=_COLOR_PERCENTILE, linestyle=":", linewidth=1.3, label=f"p90: ${p90_3:.1f}")
    
    ax3.set_title("Chi tiêu mỗi lượt mua (Spend / Trip - Log scale)", fontsize=11, fontweight="bold", color="#1F2937")
    ax3.set_xlabel("Spend per trip ($)", fontsize=9.5)
    ax3.set_ylabel("Số household", fontsize=9.5)
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"${int(y):,}"))
    ax3.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="#F8FAFC")
    ax3.text(0.03, 0.92, f"Loại bỏ (null/short): {n_excluded3}", transform=ax3.transAxes,
             fontsize=8.5, color="#64748B", bbox=dict(boxstyle="round,pad=0.2", facecolor="#F1F5F9", edgecolor="none"))
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel 4: Median IPT (Integer Bins 1..30 and '30+')
    # -------------------------------------------------------------
    ax4 = axes_flat[3]
    s4 = data["median_ipt"].dropna()
    n_excluded4 = total_hh - len(s4)
    
    # Binning integer 1..30 and 30+
    ipt_counts = {}
    for d in range(1, 31):
        ipt_counts[str(d)] = (np.floor(s4) == d).sum()
    ipt_counts["30+"] = (s4 > 30).sum()
    
    x4 = np.arange(len(ipt_counts))
    labels4 = list(ipt_counts.keys())
    counts4 = list(ipt_counts.values())
    
    colors4 = ["#1E40AF" if k == "7" else _COLOR_BAR for k in labels4]
    ax4.bar(x4, counts4, color=colors4, width=0.75)
    
    # Show labels every 5 days
    tick_indices = [0, 4, 6, 9, 14, 19, 24, 29, 30]
    ax4.set_xticks(tick_indices)
    ax4.set_xticklabels([labels4[i] for i in tick_indices], fontsize=8.5)
    
    med4 = s4.median()
    p10_4, p90_4 = np.percentile(s4, [10, 90])
    
    # Reference line at Day 7 (index 6)
    ax4.axvline(6, color=_COLOR_ACCENT, linestyle="--", linewidth=1.8, label="Vạch mốc 7 ngày (H=7)")
    
    ax4.set_title("Nhịp mua sắm hộ gia đình (Median IPT)", fontsize=11, fontweight="bold", color="#1F2937")
    ax4.set_xlabel("Median khoảng cách giữa các lần mua (ngày)", fontsize=9.5)
    ax4.set_ylabel("Số household", fontsize=9.5)
    ax4.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="#F8FAFC")
    ax4.text(0.03, 0.92, f"Median: {med4:.1f}d | Loại: {n_excluded4}", transform=ax4.transAxes,
             fontsize=8.5, color="#64748B", bbox=dict(boxstyle="round,pad=0.2", facecolor="#F1F5F9", edgecolor="none"))
    ax4.grid(axis="y", linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel 5: Regularity Index CV of IPT (with refs at 0.5 and 1.0)
    # -------------------------------------------------------------
    ax5 = axes_flat[4]
    s5 = data["cv_ipt"].dropna()
    n_excluded5 = total_hh - len(s5)
    
    max_cv = min(3.0, np.percentile(s5, 99) * 1.1)
    bins5 = np.linspace(0, max_cv, 35)
    ax5.hist(s5, bins=bins5, color=_COLOR_BAR, edgecolor="white", alpha=0.85)
    
    med5 = s5.median()
    p10_5, p90_5 = np.percentile(s5, [10, 90])
    
    ax5.axvline(med5, color=_COLOR_MEDIAN, linestyle="-", linewidth=1.8, label=f"Median: {med5:.2f}")
    ax5.axvline(0.5, color=_COLOR_REF, linestyle="--", linewidth=1.4, label="Đều đặn (CV=0.5)")
    ax5.axvline(1.0, color=_COLOR_ACCENT, linestyle="--", linewidth=1.4, label="Bất định (CV=1.0)")
    
    ax5.set_title("Độ biến động nhịp mua (CV = Std / Mean IPT)", fontsize=11, fontweight="bold", color="#1F2937")
    ax5.set_xlabel("Hệ số biến thiên IPT (CV)", fontsize=9.5)
    ax5.set_ylabel("Số household", fontsize=9.5)
    ax5.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="#F8FAFC")
    ax5.text(0.03, 0.92, f"Loại (<3 ngày mua): {n_excluded5}", transform=ax5.transAxes,
             fontsize=8.5, color="#64748B", bbox=dict(boxstyle="round,pad=0.2", facecolor="#F1F5F9", edgecolor="none"))
    ax5.grid(axis="y", linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel 6: Executive Scorecard / Metadata
    # -------------------------------------------------------------
    ax6 = axes_flat[5]
    ax6.axis("off")
    
    pct_short = (summary["short_span"].mean()) * 100
    card_text = (
        "TỔNG QUAN CHỈ SỐ KHÁCH HÀNG\n"
        "───────────────────────────────\n"
        f"• Tổng số household: {total_hh:,}\n"
        f"• Hộ có span < 28 ngày: {summary['short_span'].sum():,} ({pct_short:.1f}%)\n"
        f"• Trung vị F (ngày/tuần): {summary['F'].median():.2f}\n"
        f"• Trung vị Spend/tuần: ${summary['spend_per_week'].median():.1f}\n"
        f"• Trung vị Spend/trip: ${summary['spend_per_trip'].median():.1f}\n"
        f"• Trung vị Median IPT: {summary['median_ipt'].median():.1f} ngày\n"
        f"• Trung vị CV IPT: {summary['cv_ipt'].median():.2f}\n"
        "───────────────────────────────\n"
        "Nhận xét chính:\n"
        "1. Đa số hộ đi chợ 1-2 lần/tuần (F ≈ 0.8).\n"
        "2. Spend per week lệch phải nặng (power-law).\n"
        "3. CV ≈ 0.8: Phần lớn khách hàng có thói\n"
        "   quen mua sắm tương đối đều đặn."
    )
    ax6.text(
        0.1,
        0.5,
        card_text,
        transform=ax6.transAxes,
        fontsize=9.5,
        fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.9),
    )

    fig.suptitle(
        "Customer Overview: Phân Phối Tần Suất (F), Giá Trị Chi Tiêu (M) và Nhịp Mua Sắm (IPT)",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
        y=0.98,
    )
    plt.tight_layout()
    return fig


def plot_customer_lorenz_curves(summary: pd.DataFrame) -> Figure:
    """Plot dual-perspective concentration analysis: Lorenz Curve and Pareto Top Curve.

    Panel 1 (Left): Standard economic Lorenz Curve (ascending order from lowest spend to highest).
    Panel 2 (Right): Retail/CRM Pareto Concentration Curve (descending order from highest spend down).

    Parameters
    ----------
    summary : pd.DataFrame
        Output of build_household_summary.

    Returns
    -------
    Figure
        Matplotlib figure containing the 2-panel concentration comparison.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=150)

    # -------------------------------------------------------------
    # Panel 1: Standard Lorenz Curve (Ascending from low to high)
    # -------------------------------------------------------------
    p_spend, L_spend = lorenz_curve(summary["total_spend"])
    g_spend = gini(summary["total_spend"])

    p_days, L_days = lorenz_curve(summary["n_purchase_days"])
    g_days = gini(summary["n_purchase_days"])

    # 45-degree line of perfect equality
    ax1.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", linewidth=1.5, label="Đường bình đẳng tuyệt đối (Gini = 0)")
    ax1.plot(p_spend, L_spend, color=_COLOR_PRIMARY, linewidth=2.3, label=f"Tổng chi tiêu (Total Spend) — Gini = {g_spend:.3f}")
    ax1.plot(p_days, L_days, color="#059669", linewidth=2.0, linestyle="-.", label=f"Số ngày mua (Purchase Days) — Gini = {g_days:.3f}")

    # Point at bottom 80%
    idx_80 = int(np.searchsorted(p_spend, 0.80))
    val_bottom_80 = L_spend[idx_80]
    pct_top_20 = (1.0 - val_bottom_80) * 100.0

    ax1.scatter([0.80], [val_bottom_80], color=_COLOR_ACCENT, s=45, zorder=5)
    ax1.axvline(0.80, color="#CBD5E1", linestyle=":", linewidth=1.2)
    ax1.annotate(
        f"Mốc 80% hộ chi tiêu thấp nhất:\n• Chiếm {val_bottom_80*100:.1f}% tổng doanh thu\n• (20% hộ cao nhất còn lại chiếm {pct_top_20:.1f}%)",
        xy=(0.80, val_bottom_80),
        xytext=(0.28, 0.58),
        arrowprops=dict(arrowstyle="->", color=_COLOR_ACCENT, lw=1.5),
        fontsize=8.5,
        fontweight="bold",
        color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF2F2", edgecolor="#FCA5A5", alpha=0.95),
    )

    ax1.set_title("1. Đường cong Lorenz (Chuẩn Kinh Tế Học: Thấp → Cao)", fontsize=11, fontweight="bold", pad=12)
    ax1.set_xlabel("Tỷ lệ tích lũy hộ gia đình (Sắp xếp TĂNG DẦN từ chi tiêu thấp nhất)", fontsize=9.5)
    ax1.set_ylabel("Tỷ lệ tích lũy doanh thu", fontsize=9.5)
    ax1.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.02)
    ax1.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")
    ax1.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")

    # -------------------------------------------------------------
    # Panel 2: Pareto Concentration Curve (Descending from Top VIP down)
    # -------------------------------------------------------------
    spend_desc = np.sort(summary["total_spend"].dropna().values)[::-1]
    cum_spend_desc = np.cumsum(spend_desc) / np.sum(spend_desc)
    p_desc = np.arange(1, len(spend_desc) + 1) / len(spend_desc)
    p_desc = np.insert(p_desc, 0, 0.0)
    cum_spend_desc = np.insert(cum_spend_desc, 0, 0.0)

    days_desc = np.sort(summary["n_purchase_days"].dropna().values)[::-1]
    cum_days_desc = np.cumsum(days_desc) / np.sum(days_desc)
    p_days_desc = np.arange(1, len(days_desc) + 1) / len(days_desc)
    p_days_desc = np.insert(p_days_desc, 0, 0.0)
    cum_days_desc = np.insert(cum_days_desc, 0, 0.0)

    # Reference lines for 50% and 80% revenue
    ax2.axhline(0.50, color="#94A3B8", linestyle=":", linewidth=1.2)
    ax2.axhline(0.80, color="#94A3B8", linestyle=":", linewidth=1.2)
    ax2.axvline(0.20, color="#94A3B8", linestyle=":", linewidth=1.2)

    ax2.plot(p_desc, cum_spend_desc, color=_COLOR_PRIMARY, linewidth=2.5, label="Doanh thu tích lũy từ Top hộ")
    ax2.plot(p_days_desc, cum_days_desc, color="#059669", linewidth=2.0, linestyle="-.", label="Ngày mua tích lũy từ Top hộ")

    # Point at Top 20%
    val_top_20 = cum_spend_desc[int(0.20 * len(spend_desc))]
    ax2.scatter([0.20], [val_top_20], color=_COLOR_PRIMARY, s=55, zorder=5)
    ax2.annotate(
        f"TOP 20% HỘ DẪN ĐẦU:\nChiếm {val_top_20*100:.1f}% tổng doanh thu",
        xy=(0.20, val_top_20),
        xytext=(0.35, 0.42),
        arrowprops=dict(arrowstyle="->", color=_COLOR_PRIMARY, lw=1.5),
        fontsize=9,
        fontweight="bold",
        color="#1E3A8A",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#EFF6FF", edgecolor="#93C5FD", alpha=0.95),
    )

    # Point at 50% revenue
    idx_50 = int(np.searchsorted(cum_spend_desc, 0.50))
    pct_hh_50 = p_desc[idx_50] * 100.0
    ax2.scatter([p_desc[idx_50]], [0.50], color=_COLOR_ACCENT, s=40, zorder=5)
    ax2.annotate(
        f"50% doanh thu chỉ do\n{pct_hh_50:.1f}% số hộ tạo ra",
        xy=(p_desc[idx_50], 0.50),
        xytext=(p_desc[idx_50] + 0.12, 0.62),
        arrowprops=dict(arrowstyle="->", color=_COLOR_ACCENT, lw=1.3),
        fontsize=8.5,
        fontweight="bold",
        color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF2F2", edgecolor="#FCA5A5", alpha=0.95),
    )

    ax2.set_title("2. Đường cong Tích Lũy Pareto (Chuẩn Bán Lẻ / CRM: Cao → Thấp)", fontsize=11, fontweight="bold", pad=12)
    ax2.set_xlabel("Tỷ lệ Top hộ gia đình (Sắp xếp GIẢM DẦN từ VIP chi tiêu nhiều nhất)", fontsize=9.5)
    ax2.set_ylabel("Tỷ lệ tích lũy doanh thu đóng góp", fontsize=9.5)
    ax2.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.02)
    ax2.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")
    ax2.legend(loc="lower right", fontsize=8.5, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")

    plt.tight_layout()
    return fig

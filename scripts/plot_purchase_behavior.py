"""
scripts/plot_purchase_behavior.py
─────────────────────────────────
Phân tích hành vi mua lại sâu hơn (Dunnhumby).

Hàm public:
  load_behavior_data(data_path)              → (hh_counts, intervals)
  plot_one_and_done(hh_counts, path)         → barh: One-and-Done vs Repeaters
  plot_interpurchase_histogram(intervals, path) → histogram interval + percentiles
  plot_cumulative_return_rate(intervals, path)  → CDF return rate theo H ngày

Màu: xanh nhạt đơn sắc #64B5F6.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

_COLOR = "#64B5F6"
_EDGE  = "#1565C0"
_DARK  = "#0D47A1"
_LINE  = "#1565C0"
_RED   = "#D32F2F"

ONESHOT_BINS   = [0, 1, 5, 12, 39, 79, 149, float("inf")]
ONESHOT_LABELS = [
    "1 lần  (one-and-done)",
    "2–5 lần",
    "6–12 lần",
    "13–39 lần",
    "40–79 lần",
    "80–149 lần",
    "150+ lần",
]


# ── Load ─────────────────────────────────────────────────────────────────────

def load_behavior_data(
    data_path: Path | str,
) -> tuple[pd.Series, np.ndarray]:
    """
    Đọc transaction_data, tính:
      hh_counts  : Series[household_key → số basket duy nhất]
      intervals  : ndarray số ngày giữa 2 ngày mua sắm riêng biệt liên tiếp (distinct shopping days, gap >= 1)
    """
    df = pd.read_csv(
        Path(data_path),
        usecols=["household_key", "BASKET_ID", "DAY"],
        dtype={"household_key": str, "BASKET_ID": str},
    )

    # 1. Số basket / household
    hh_counts = df.groupby("household_key")["BASKET_ID"].nunique()

    # 2. Khoảng cách giữa các NGÀY mua sắm riêng biệt (distinct shopping days)
    # Bỏ qua các giao dịch cùng ngày (gap = 0)
    day_df = (
        df[["household_key", "DAY"]]
        .drop_duplicates()
        .sort_values(["household_key", "DAY"])
    )
    day_df["prev"] = day_df.groupby("household_key")["DAY"].shift(1)
    day_df["gap"]  = day_df["DAY"] - day_df["prev"]
    intervals = day_df["gap"].dropna().astype(int).values

    return hh_counts, intervals


# ── Helper ────────────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Saved: {output_path}")


# ── Chart 1: One-and-Done vs Repeaters ───────────────────────────────────────

def plot_one_and_done(
    hh_counts: pd.Series,
    output_path: Path | str,
) -> None:
    """
    Horizontal bar chart: % HH trong từng bin purchase count.
    Ranh giới One-and-Done (1 lần) vs Repeaters (≥2 lần) được đánh dấu rõ.
    """
    cut    = pd.cut(hh_counts, bins=ONESHOT_BINS,
                    labels=ONESHOT_LABELS, right=True)
    counts = cut.value_counts(sort=False).reindex(ONESHOT_LABELS).fillna(0)
    pct    = (counts / counts.sum() * 100).tolist()

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#FFFFFF")
    bars = ax.barh(ONESHOT_LABELS, pct,
                   color=_COLOR, edgecolor=_EDGE, linewidth=0.6, height=0.52)

    x_max = max(pct) if pct else 1
    for bar, val in zip(bars, pct):
        ax.text(
            bar.get_width() + x_max * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left",
            fontsize=10, color=_DARK,
        )

    # ── Ranh giới One-and-Done vs Repeaters ──────────────────────────────
    # y=0.5 nằm giữa bar[0] ("1 lần") và bar[1] ("2-5 lần"); gọi trước invert
    ax.axhline(y=0.5, color=_RED, linestyle="--", linewidth=1.6, alpha=0.85, zorder=5)
    ax.text(x_max * 0.55, 0.22,
            "One-and-Done  ↑",
            fontsize=8.5, color=_RED, style="italic", ha="center", va="center",
            bbox={"fc": "white", "ec": _RED, "alpha": 0.75, "boxstyle": "round,pad=0.25"})
    ax.text(x_max * 0.55, 0.78,
            "↓  Repeaters (≥ 2 lần)",
            fontsize=8.5, color=_LINE, style="italic", ha="center", va="center",
            bbox={"fc": "white", "ec": _LINE, "alpha": 0.75, "boxstyle": "round,pad=0.25"})

    ax.invert_yaxis()
    ax.set_title("Tỷ lệ Household theo Số Lần Mua — One-and-Done vs Repeaters",
                 fontsize=12, fontweight="bold", color=_DARK, pad=10)
    ax.set_xlabel("% Household", fontsize=10, color=_LINE)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_xlim(right=x_max * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#F8FBFF")
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    _save_fig(fig, Path(output_path))


# ── Chart 2: Interpurchase Interval Histogram ─────────────────────────────────

def plot_interpurchase_histogram(
    intervals: np.ndarray,
    output_path: Path | str,
    clip_days: int = 30,
) -> None:
    """
    Histogram phân phối khoảng cách giữa các ngày mua riêng biệt (gap >= 1 ngày).
    Cắt đuôi tại clip_days (mặc định 30 ngày) để zoom vào vùng đầu.
    Cột đầu (gap = 1 ngày) dính liền trục Y (xlim left=0.5).
    Đã lược bỏ các đường phân vị P để đồ thị thoáng và tập trung vào hình thái phân phối.
    """
    clipped = intervals[intervals <= clip_days]
    pct_shown = len(clipped) / len(intervals) * 100

    fig, ax = plt.subplots(figsize=(11, 5), facecolor="#FFFFFF")

    # Bins 1-ngày căn giữa: [0.5, 1.5), [1.5, 2.5), ...
    bin_edges = np.arange(0.5, clip_days + 1.5, 1)
    ax.hist(
        clipped,
        bins=bin_edges,
        color=_COLOR,
        edgecolor=_EDGE,
        linewidth=0.5,
        alpha=0.88,
    )

    ax.set_title(
        f"Phân phối Khoảng cách giữa các Ngày Mua Riêng biệt (Interpurchase Interval)\n"
        f"Zoom vùng đầu ≤{clip_days} ngày ({pct_shown:.1f}% tổng lượt mua sắm khác ngày)",
        fontsize=12,
        fontweight="bold",
        color=_DARK,
        pad=10,
    )
    ax.set_xlabel("Số ngày giữa 2 ngày mua sắm liên tiếp (ngày)", fontsize=10, color=_LINE)
    ax.set_ylabel("Tần số (số lượt)", fontsize=10, color=_LINE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(left=0.5, right=clip_days + 0.5)
    ax.set_xticks([1, 2, 4, 6, 8, 10, 15, 20, 25, 30])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#F8FBFF")
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    _save_fig(fig, Path(output_path))


# ── Chart 3: Cumulative Return Rate ───────────────────────────────────────────

def plot_cumulative_return_rate(
    intervals: np.ndarray,
    output_path: Path | str,
    horizons: tuple[int, ...] = (7, 14, 30, 60, 90),
) -> None:
    """
    Đường tích lũy: % lượt mua quay lại trong vòng H ngày (CDF of intervals).
    Dán nhãn tại H = 7, 14, 30, 60, 90.
    """
    h_max  = max(horizons)
    h_range = np.arange(1, h_max + 1)
    cdf     = np.array([(intervals <= h).mean() * 100 for h in h_range])

    fig, ax = plt.subplots(figsize=(11, 5), facecolor="#FFFFFF")

    ax.fill_between(h_range, cdf, alpha=0.14, color=_COLOR)
    ax.plot(h_range, cdf, color=_LINE, linewidth=2.2, zorder=4)

    # Nhãn tại các horizon cụ thể
    # Xen kẽ label lên/xuống tránh đè nhau
    offsets = [+6, -6, +6, -6, +6]
    for h, dy in zip(horizons, offsets):
        val = float((intervals <= h).mean() * 100)
        ax.vlines(x=h, ymin=0, ymax=val,
                  colors=_LINE, linestyle=":", linewidth=1.0, alpha=0.5)
        ax.scatter(h, val, color=_LINE, s=55, zorder=6)
        ax.annotate(
            f"H={h}d\n{val:.1f}%",
            xy=(h, val), xytext=(h, val + dy),
            ha="center", va="bottom" if dy > 0 else "top",
            fontsize=8.5, color=_DARK,
            arrowprops={"arrowstyle": "-", "color": _DARK, "lw": 0.7},
            bbox={"boxstyle": "round,pad=0.3", "fc": "white",
                      "ec": _LINE, "alpha": 0.85, "linewidth": 0.8},
        )

    ax.set_title(
        "Cumulative Return Rate theo Horizon H\n"
        "% lượt mua được theo sau bởi lần mua tiếp theo trong vòng H ngày",
        fontsize=12, fontweight="bold", color=_DARK, pad=10,
    )
    ax.set_xlabel("Horizon H (ngày)", fontsize=10, color=_LINE)
    ax.set_ylabel("% Lượt Quay lại Tích lũy", fontsize=10, color=_LINE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_xlim(1, h_max)
    ax.set_ylim(0, 107)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#F8FBFF")
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    _save_fig(fig, Path(output_path))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/raw/Dunnhumby/transaction_data.csv")
    parser.add_argument("--output-dir", default="output/charts")
    args = parser.parse_args()

    out = Path(args.output_dir)
    print(f"Đọc: {args.data} …")
    hh_counts, intervals = load_behavior_data(args.data)
    print(f"  {len(hh_counts):,} HH | {len(intervals):,} intervals | "
          f"P50={int(np.percentile(intervals,50))}d P90={int(np.percentile(intervals,90))}d")

    print("Chart 1: one-and-done …")
    plot_one_and_done(hh_counts, out / "chart_behavior_oneshot.png")

    print("Chart 2: interpurchase histogram …")
    plot_interpurchase_histogram(intervals, out / "chart_behavior_intervals.png")

    print("Chart 3: cumulative return rate …")
    plot_cumulative_return_rate(intervals, out / "chart_behavior_return_rate.png")

    print("Hoàn thành!")


if __name__ == "__main__":
    main()

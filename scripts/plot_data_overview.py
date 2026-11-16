"""
scripts/plot_data_overview.py
─────────────────────────────
Phân tích tổng quan thời gian hoạt động của dataset Dunnhumby.

Hàm public:
  load_temporal_data(data_path)              → raw DataFrame (household_key, BASKET_ID, DAY)
  plot_transactions_over_time(df, path)      → line chart: lượt giao dịch theo tuần
  plot_household_lifespan(df, path)          → barh: phân bin tuổi thọ HH
  plot_cohort_heatmap(df, path)              → heatmap: % cohort hoạt động theo tháng
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

_BLUE_LINE   = "#1565C0"
_BLUE_FILL   = "#64B5F6"
_BLUE_BAR    = "#64B5F6"
_BLUE_EDGE   = "#1565C0"
_CMAP        = "Blues"

LIFESPAN_BINS   = [0, 90, 180, 300, 450, 600, float("inf")]
LIFESPAN_LABELS = ["≤90 ngày", "91–180", "181–300", "301–450", "451–600", "601+"]


# ── Load ─────────────────────────────────────────────────────────────────────

def load_temporal_data(data_path: Path | str) -> pd.DataFrame:
    """
    Đọc 3 cột cần thiết cho phân tích thời gian:
      household_key | BASKET_ID | DAY
    """
    return pd.read_csv(
        Path(data_path),
        usecols=["household_key", "BASKET_ID", "DAY"],
        dtype={"household_key": str, "BASKET_ID": str},
    )


# ── Shared ────────────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Saved: {output_path}")


# ── Chart 1: Lượt giao dịch theo tuần ───────────────────────────────────────

def plot_transactions_over_time(
    df: pd.DataFrame,
    output_path: Path | str,
) -> None:
    """
    Line chart: số basket (lượt mua) duy nhất mỗi tuần.
    Cho thấy độ dài dataset, xu hướng, và temporal gap.
    """
    df = df.copy()
    df["week"] = (df["DAY"] - 1) // 7 + 1          # week 1..102
    weekly = (
        df.groupby("week")["BASKET_ID"]
        .nunique()
        .rename("n_baskets")
    )

    # Rolling average 4 tuần để làm trơn
    rolling = weekly.rolling(4, center=True).mean()

    fig, ax = plt.subplots(figsize=(13, 4.5), facecolor="#FFFFFF")

    # Vùng fill + đường daily
    ax.fill_between(weekly.index, weekly.values,
                    alpha=0.18, color=_BLUE_FILL)
    ax.plot(weekly.index, weekly.values,
            color=_BLUE_FILL, linewidth=0.9, alpha=0.6)
    # Đường rolling avg nổi bật
    ax.plot(rolling.index, rolling.values,
            color=_BLUE_LINE, linewidth=2.0, label="4-tuần avg")

    # Nhãn tháng trên x-axis (mỗi ~4 tuần)
    month_ticks = list(range(1, int(weekly.index.max()) + 1, 4))
    month_labels = [f"T{(w - 1) // 4 + 1}" for w in month_ticks]
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels, fontsize=8)

    ax.set_title("Lượt Giao dịch (Basket) theo Tuần", fontsize=13,
                 fontweight="bold", color="#0D47A1", pad=10)
    ax.set_xlabel("Thời gian (tháng tương đối)", fontsize=10, color=_BLUE_LINE)
    ax.set_ylabel("Số Basket / Tuần", fontsize=10, color=_BLUE_LINE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#F8FBFF")
    ax.legend(fontsize=9, framealpha=0.4)
    fig.tight_layout()
    _save_fig(fig, Path(output_path))


# ── Chart 2: Phân phối thời gian hoạt động của Household ──────────────────────

def plot_household_lifespan(
    df: pd.DataFrame,
    output_path: Path | str,
    bin_width: int = 20,
) -> None:
    """
    Histogram: Phân phối thời gian hoạt động của Household (Activity Span).
    Thời gian hoạt động = max(DAY) - min(DAY) + 1 của mỗi HH.
    """
    span = (
        df.groupby("household_key")["DAY"]
        .agg(lambda x: int(x.max() - x.min() + 1))
        .rename("activity_days")
    )
    median_span = float(span.median())

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#FFFFFF")

    max_day = int(df["DAY"].max())
    bin_edges = np.arange(0, max_day + bin_width + 1, bin_width)
    ax.hist(
        span,
        bins=bin_edges,
        color=_BLUE_BAR,
        edgecolor=_BLUE_EDGE,
        linewidth=0.5,
        alpha=0.88,
    )

    ax.set_title(
        f"Phân phối Thời gian Hoạt động của Household (Activity Span)\n"
        f"Thời gian hoạt động = Ngày mua cuối − Ngày mua đầu + 1 (Median: {median_span:.0f} ngày)",
        fontsize=12,
        fontweight="bold",
        color="#0D47A1",
        pad=10,
    )
    ax.set_xlabel("Thời gian hoạt động (ngày)", fontsize=10, color=_BLUE_LINE)
    ax.set_ylabel("Số lượng Household", fontsize=10, color=_BLUE_LINE)
    ax.set_xlim(left=0, right=max_day + bin_width)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}d"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#F8FBFF")
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    _save_fig(fig, Path(output_path))


# ── Chart 3: Cohort Heatmap ───────────────────────────────────────────────────

def plot_cohort_heatmap(
    df: pd.DataFrame,
    output_path: Path | str,
    period: str = "bimonth",   # 'month' | 'bimonth' | 'quarter'
) -> None:
    """
    Heatmap: % mỗi cohort (nhóm gia nhập) còn hoạt động ở từng period offset.

    Cohort = period mà HH xuất hiện lần đầu.
    Offset = số period kể từ khi gia nhập.
    Màu: Blues — đậm hơn = tỷ lệ active cao hơn.
    """
    df = df.copy()

    # Quy đổi DAY → period index
    period_days = {"month": 30, "bimonth": 60, "quarter": 90}
    unit        = period_days.get(period, 60)
    period_name = {"month": "Tháng", "bimonth": "2 Tháng", "quarter": "Quý"}[period]

    df["period"] = (df["DAY"] - 1) // unit

    # Cohort = period đầu tiên của mỗi HH
    first = df.groupby("household_key")["period"].min().rename("cohort")
    df    = df.join(first, on="household_key")
    df["offset"] = df["period"] - df["cohort"]

    max_offset = df["offset"].max()

    # Ma trận: cohort × offset → số HH duy nhất còn active
    active = (
        df.groupby(["cohort", "offset"])["household_key"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=range(max_offset + 1), fill_value=0)
    )
    cohort_size = first.value_counts().sort_index()
    pct_matrix  = active.div(cohort_size, axis=0).multiply(100)

    # Mask các ô "tương lai" (offset > total_periods - cohort)
    total_periods = int(df["period"].max())
    mask = np.zeros(pct_matrix.shape, dtype=bool)
    for i, cohort_idx in enumerate(pct_matrix.index):
        max_valid_offset = total_periods - int(cohort_idx)
        mask[i, max_valid_offset + 1:] = True
    pct_masked = np.where(mask, np.nan, pct_matrix.values)

    # ── Vẽ ──────────────────────────────────────────────────────────────────
    n_cohorts, n_offsets = pct_matrix.shape
    fig_w = max(12, n_offsets * 0.55)
    fig_h = max(5,  n_cohorts * 0.42)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="#FFFFFF")

    cmap = plt.get_cmap(_CMAP).copy()
    cmap.set_bad(color="#EEEEEE")          # ô masked → xám nhạt

    im = ax.imshow(pct_masked, aspect="auto", cmap=cmap,
                   vmin=0, vmax=100, interpolation="nearest")

    # Giá trị trong ô
    for r in range(n_cohorts):
        for c in range(n_offsets):
            val = pct_masked[r, c]
            if not np.isnan(val):
                color = "white" if val > 60 else "#1A237E"
                ax.text(c, r, f"{val:.0f}%", ha="center", va="center",
                        fontsize=7, color=color)

    # Trục
    ax.set_xticks(range(n_offsets))
    ax.set_xticklabels([f"+{c}" for c in range(n_offsets)], fontsize=8)
    ax.set_yticks(range(n_cohorts))
    ax.set_yticklabels(
        [f"{period_name} {int(c) + 1}  (n={int(cohort_size.get(c, 0))})"
         for c in pct_matrix.index],
        fontsize=8,
    )

    ax.set_title(
        f"Cohort Retention — % Household còn hoạt động\n"
        f"(theo {period_name}, offset kể từ lần mua đầu)",
        fontsize=12, fontweight="bold", color="#0D47A1", pad=12,
    )
    ax.set_xlabel(f"Offset ({period_name} kể từ gia nhập)", fontsize=10, color=_BLUE_LINE)
    ax.set_ylabel("Cohort (nhóm gia nhập)", fontsize=10, color=_BLUE_LINE)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("% Active", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

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
    df = load_temporal_data(args.data)
    print(f"  {len(df):,} rows | DAY: {df['DAY'].min()}–{df['DAY'].max()}")

    print("Chart 1: transactions over time …")
    plot_transactions_over_time(df, out / "chart_temporal_transactions.png")

    print("Chart 2: household lifespan …")
    plot_household_lifespan(df, out / "chart_temporal_lifespan.png")

    print("Chart 3: cohort heatmap …")
    plot_cohort_heatmap(df, out / "chart_temporal_cohort.png")

    print("Hoàn thành!")


if __name__ == "__main__":
    main()

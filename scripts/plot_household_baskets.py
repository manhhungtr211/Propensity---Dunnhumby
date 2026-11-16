"""
scripts/plot_household_baskets.py
─────────────────────────────────
Phân tích tần suất mua lại của household (Dunnhumby).

Hàm public:
  load_hh_metrics(data_path)                    → DataFrame cấp HH
  plot_purchase_frequency(hh, path)             → % HH theo nhóm
  plot_spending_by_frequency(hh, path)          → % Spending theo nhóm
  plot_baskets_per_hh(hh, path)                 → avg basket/HH theo nhóm
  plot_active_days(hh, path)                    → avg active days/HH theo nhóm
  plot_purchase_rate_weekly(hh, path)           → avg basket/tuần theo nhóm
  plot_purchase_rate_monthly(hh, path)          → avg basket/tháng theo nhóm
  summary_table(hh)                             → DataFrame tổng hợp

Bin:
  Mua ít         1–12 lần
  Thỉnh thoảng  13–39 lần
  Hàng tháng    40–79 lần
  Hàng tuần     80–149 lần
  Thường xuyên  150+ lần

Màu: xanh nhạt đơn sắc #64B5F6 — highlight sau nếu cần.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
BINS   = [0, 12, 39, 79, 149, float("inf")]
LABELS = ["Mua ít\n(1–12)", "Thỉnh thoảng\n(13–39)",
          "Hàng tháng\n(40–79)", "Hàng tuần\n(80–149)", "Thường xuyên\n(150+)"]
_COLOR = "#64B5F6"    # blue-300, xanh nhạt vừa
_EDGE  = "#1565C0"    # blue-800, viền


# ── Load ─────────────────────────────────────────────────────────────────────

def load_hh_metrics(data_path: Path | str) -> pd.DataFrame:
    """
    Đọc transaction_data, trả về DataFrame cấp household:
      household_key | purchase_count | total_spending | active_days
      | rate_weekly | rate_monthly | freq_bin

    Ghi chú:
      active_days  = số ngày (DAY) duy nhất HH có giao dịch
      rate_weekly  = purchase_count / (total_dataset_days / 7)
      rate_monthly = purchase_count / (total_dataset_days / 30)
    """
    df = pd.read_csv(
        Path(data_path),
        usecols=["household_key", "BASKET_ID", "SALES_VALUE", "DAY"],
        dtype={"household_key": str, "BASKET_ID": str},
    )

    total_days = df["DAY"].max()   # span toàn dataset (DAY là ordinal)

    hh = (
        df.groupby("household_key")
        .agg(
            purchase_count=("BASKET_ID",    "nunique"),
            total_spending=("SALES_VALUE",  "sum"),
            active_days   =("DAY",          "nunique"),
        )
        .reset_index()
    )

    hh["rate_weekly"]  = (hh["purchase_count"] / (total_days / 7)).round(2)
    hh["rate_monthly"] = (hh["purchase_count"] / (total_days / 30)).round(2)
    hh["freq_bin"] = pd.cut(
        hh["purchase_count"], bins=BINS, labels=LABELS, right=True,
    )
    return hh


# ── Internal helpers ──────────────────────────────────────────────────────────

def _base_barh(
    ax: plt.Axes,
    values: list[float],
    labels_on_bar: list[str],
    title: str,
    xlabel: str,
) -> None:
    """Horizontal bar chart dùng chung cho mọi metric."""
    bars = ax.barh(
        LABELS, values,
        color=_COLOR, edgecolor=_EDGE, linewidth=0.6, height=0.52,
    )
    x_max = max(values) if values else 1
    for bar, lbl in zip(bars, labels_on_bar):
        ax.text(
            bar.get_width() + x_max * 0.015,
            bar.get_y() + bar.get_height() / 2,
            lbl, va="center", ha="left",
            fontsize=10, color="#0D47A1",
        )
    ax.set_title(title, fontsize=12, fontweight="bold", color="#0D47A1", pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color="#1565C0")
    ax.set_xlim(right=x_max * 1.22)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#F8FBFF")
    ax.tick_params(axis="both", labelsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.1f}"))


def _avg_by_group(hh: pd.DataFrame, col: str) -> pd.Series:
    return (
        hh.groupby("freq_bin", observed=True)[col]
        .mean()
        .reindex(LABELS)
        .fillna(0)
    )


def _save_fig(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Saved: {output_path}")


def _make_fig(title, xlabel, values, fmt="{:.1f}") -> tuple:
    labels_on_bar = [fmt.format(v) for v in values]
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#FFFFFF")
    _base_barh(ax, values, labels_on_bar, title, xlabel)
    fig.tight_layout()
    return fig, ax


# ── Chart 1 — % Household ────────────────────────────────────────────────────

def plot_purchase_frequency(hh: pd.DataFrame, output_path: Path | str) -> None:
    """% Household trong mỗi nhóm tần suất."""
    counts = hh["freq_bin"].value_counts(sort=False).reindex(LABELS).fillna(0)
    pct    = (counts / counts.sum() * 100).tolist()
    fig, _ = _make_fig("% Household theo Tần suất Mua", "% Household",
                        pct, fmt="{:.1f}%")
    _save_fig(fig, Path(output_path))


# ── Chart 2 — % Spending ─────────────────────────────────────────────────────

def plot_spending_by_frequency(hh: pd.DataFrame, output_path: Path | str) -> None:
    """% tổng Spending trong mỗi nhóm tần suất."""
    spend = (
        hh.groupby("freq_bin", observed=True)["total_spending"]
        .sum().reindex(LABELS).fillna(0)
    )
    pct = (spend / spend.sum() * 100).tolist()
    fig, _ = _make_fig("% Spending theo Tần suất Mua", "% Tổng Spending",
                        pct, fmt="{:.1f}%")
    _save_fig(fig, Path(output_path))


# ── Chart 3 — Avg baskets / HH ───────────────────────────────────────────────

def plot_baskets_per_hh(hh: pd.DataFrame, output_path: Path | str) -> None:
    """Trung bình số basket mỗi HH trong từng nhóm."""
    vals = _avg_by_group(hh, "purchase_count").tolist()
    fig, _ = _make_fig("TB Số Basket / Household theo Nhóm",
                        "Avg Basket / HH", vals, fmt="{:.0f}")
    _save_fig(fig, Path(output_path))


# ── Chart 4 — Avg active days / HH ───────────────────────────────────────────

def plot_active_days(hh: pd.DataFrame, output_path: Path | str) -> None:
    """Trung bình số ngày có giao dịch (active days) mỗi HH trong từng nhóm."""
    vals = _avg_by_group(hh, "active_days").tolist()
    fig, _ = _make_fig("TB Active Days / Household theo Nhóm",
                        "Avg Active Days / HH", vals, fmt="{:.0f}")
    _save_fig(fig, Path(output_path))


# ── Chart 5 — Avg purchase rate / week ───────────────────────────────────────

def plot_purchase_rate_weekly(hh: pd.DataFrame, output_path: Path | str) -> None:
    """Trung bình số lần mua mỗi tuần (tính trên toàn span dataset) theo nhóm."""
    vals = _avg_by_group(hh, "rate_weekly").tolist()
    fig, _ = _make_fig("TB Số Lần Mua / Tuần theo Nhóm",
                        "Avg Lần Mua / Tuần", vals, fmt="{:.2f}")
    _save_fig(fig, Path(output_path))


# ── Chart 6 — Avg purchase rate / month ──────────────────────────────────────

def plot_purchase_rate_monthly(hh: pd.DataFrame, output_path: Path | str) -> None:
    """Trung bình số lần mua mỗi tháng (tính trên toàn span dataset) theo nhóm."""
    vals = _avg_by_group(hh, "rate_monthly").tolist()
    fig, _ = _make_fig("TB Số Lần Mua / Tháng theo Nhóm",
                        "Avg Lần Mua / Tháng", vals, fmt="{:.1f}")
    _save_fig(fig, Path(output_path))


# ── Summary table ─────────────────────────────────────────────────────────────

def summary_table(hh: pd.DataFrame) -> pd.DataFrame:
    """
    Trả về DataFrame tổng hợp theo nhóm freq_bin:
    n_hh | % HH | avg_baskets | avg_active_days | rate_weekly | rate_monthly
    | avg_spending_hh | % Spending
    """
    grp = (
        hh.groupby("freq_bin", observed=True)
        .agg(
            n_hh           =("household_key",  "count"),
            total_spending =("total_spending",  "sum"),
            avg_spending_hh=("total_spending",  "mean"),
            avg_baskets    =("purchase_count",  "mean"),
            avg_active_days=("active_days",     "mean"),
            avg_rate_weekly=("rate_weekly",     "mean"),
            avg_rate_monthly=("rate_monthly",   "mean"),
        )
        .reindex(LABELS).fillna(0)
    )
    grp["pct_hh"]       = (grp["n_hh"]          / grp["n_hh"].sum()           * 100).round(1)
    grp["pct_spending"] = (grp["total_spending"] / grp["total_spending"].sum() * 100).round(1)
    grp = grp.drop(columns="total_spending").reset_index()
    grp.columns = [
        "Nhóm", "Số HH", "TB Spending/HH ($)", "TB Basket/HH",
        "TB Active Days", "TB Lần/Tuần", "TB Lần/Tháng", "% HH", "% Spending",
    ]
    for col in ["TB Spending/HH ($)", "TB Basket/HH", "TB Active Days"]:
        grp[col] = grp[col].round(0).astype(int)
    grp["TB Lần/Tuần"]  = grp["TB Lần/Tuần"].round(2)
    grp["TB Lần/Tháng"] = grp["TB Lần/Tháng"].round(1)
    return grp


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/raw/Dunnhumby/transaction_data.csv")
    parser.add_argument("--output-dir", default="output/charts")
    args = parser.parse_args()

    out = Path(args.output_dir)
    print(f"Đọc dữ liệu: {args.data} …")
    hh = load_hh_metrics(args.data)
    print(f"  {len(hh):,} HH | median_baskets={hh['purchase_count'].median():.0f} "
          f"| median_active_days={hh['active_days'].median():.0f}")

    charts = [
        ("chart1_pct_household.png",    plot_purchase_frequency),
        ("chart2_pct_spending.png",     plot_spending_by_frequency),
        ("chart3_baskets_per_hh.png",   plot_baskets_per_hh),
        ("chart4_active_days.png",      plot_active_days),
        ("chart5_rate_weekly.png",      plot_purchase_rate_weekly),
        ("chart6_rate_monthly.png",     plot_purchase_rate_monthly),
    ]
    for fname, fn in charts:
        print(f"  Vẽ {fname} …")
        fn(hh, out / fname)

    print("\nBảng tổng hợp:")
    print(summary_table(hh).to_string(index=False))
    print("\nHoàn thành!")


if __name__ == "__main__":
    main()

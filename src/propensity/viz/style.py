"""Visual styling and design constants following Part 1 aesthetic and project guidelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Final
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Core Color Palette (Part 1 style)
PRIMARY: Final = "#1E40AF"     # Xanh đậm: nhóm nhấn mạnh, đường chính
BAR: Final = "#93C5FD"         # Xanh nhạt: bar mặc định
ACCENT: Final = "#DC2626"      # Đỏ: cảnh báo, nhóm dưới kỳ vọng
REF: Final = "#F59E0B"         # Cam: đường tham chiếu (H = 7, ngưỡng, rate chung)
MEDIAN: Final = "#0F172A"      # Slate đậm: đường median, kỳ vọng R x F
PERCENTILE: Final = "#64748B"  # Slate nhạt: đường p10, p90, chữ phụ
TITLE: Final = "#1F2937"       # Charcoal đậm: tiêu đề
BOX_BG: Final = "#F1F5F9"      # Nền ô chú thích bo tròn
LEGEND_BG: Final = "#F8FAFC"   # Nền khung legend
SERIES_2: Final = "#F59E0B"    # Series thứ 2 khi so 2 nhóm (ví dụ % basket và % spend)

# Slide standard dimensions
FIGSIZE_FULL: Final = (10, 5.6)
FIGSIZE_HALF: Final = (6.5, 4.6)

# Backwards compatibility aliases
BG_COLOR: Final = "#FFFFFF"
TEXT_PRIMARY: Final = TITLE
TEXT_SECONDARY: Final = PERCENTILE
GRID_COLOR: Final = "#E2E8F0"
COLOR_PRIMARY: Final = PRIMARY
COLOR_SECONDARY: Final = REF
COLOR_TERTIARY: Final = "#10B981"
CATEGORICAL_COLORS: Final = [PRIMARY, REF, "#10B981"]
GRAY_MUTED: Final = PERCENTILE
GRAY_LIGHT: Final = "#CBD5E1"
GRAY_BG_LIGHT: Final = BOX_BG
GRAY_NEUTRAL: Final = BOX_BG
COLOR_INSIGHT: Final = PRIMARY
COLOR_NON_INSIGHT: Final = BAR

# Colormaps
CMAP_SEQUENTIAL_BLUE = mcolors.LinearSegmentedColormap.from_list(
    "sequential_blue", ["#F0F7FF", "#93C5FD", "#2563EB", "#1E3A8A"]
)


def make_diverging_rate_cmap(center_rate: float, vmin: float = 0.0, vmax: float = 1.0) -> mcolors.Colormap:
    """Create a diverging colormap centered at the baseline rate (Red -> Neutral Slate -> Blue)."""
    colors = ["#EF4444", "#F87171", "#E2E8F0", "#60A5FA", "#1E40AF"]
    return mcolors.LinearSegmentedColormap.from_list("rate_diverging", colors)


def set_style() -> None:
    """Configure publication-grade matplotlib styling following Part 1 aesthetic.

    - Nền trắng; dpi=150
    - Tiêu đề 12pt đậm, màu TITLE; nhãn trục 9.5pt
    - Grid trục y nét đứt, alpha=0.5
    - Bỏ viền trên và phải
    - Bar có viền trắng (edgecolor="white"), alpha=0.85
    - Legend có khung, nền LEGEND_BG
    """
    mpl.rcParams.update(
        {
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "text.color": TITLE,
            "axes.labelcolor": TITLE,
            "xtick.color": PERCENTILE,
            "ytick.color": PERCENTILE,
            "font.family": "sans-serif",
            "font.size": 9.5,
            "axes.titlesize": 12.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.labelweight": "bold",
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#CBD5E1",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
            "legend.frameon": True,
            "legend.facecolor": LEGEND_BG,
            "legend.edgecolor": "#CBD5E1",
            "legend.fontsize": 8.5,
            "patch.edgecolor": "white",
            "figure.dpi": 150,
        }
    )


def annotate_box(
    ax: plt.Axes,
    text: str,
    loc: str = "upper left",
    coords: tuple[float, float] | None = None,
    fontsize: float = 8.5,
) -> Any:
    """Add a rounded callout box on the axes.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    text : str
        Annotation text.
    loc : str, default "upper left"
        Location ("upper left", "upper right", "lower left", "lower right").
    coords : tuple[float, float], optional
        Explicit (x, y) coordinates in axes space (0..1).
    fontsize : float, default 8.5
        Text font size.
    """
    loc_map = {
        "upper left": (0.03, 0.92, "left", "top"),
        "upper right": (0.97, 0.92, "right", "top"),
        "lower left": (0.03, 0.08, "left", "bottom"),
        "lower right": (0.97, 0.08, "right", "bottom"),
    }
    if coords is not None:
        x, y = coords
        ha, va = "left", "top"
    elif loc in loc_map:
        x, y, ha, va = loc_map[loc]
    else:
        x, y, ha, va = 0.03, 0.92, "left", "top"

    return ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        color=PERCENTILE,
        ha=ha,
        va=va,
        bbox=dict(
            boxstyle="round,pad=0.35,rounding_size=0.3",
            facecolor=BOX_BG,
            edgecolor="none",
            alpha=0.95,
        ),
        zorder=6,
    )


def add_median_lines(
    ax: plt.Axes,
    s: pd.Series | np.ndarray,
    fmt: str = ".1f",
    is_horizontal: bool = False,
) -> None:
    """Add solid median, dashed p10, and dotted p90 reference lines with formatted legend labels."""
    arr = np.asarray(s)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return

    med = float(np.median(arr))
    p10, p90 = float(np.percentile(arr, 10)), float(np.percentile(arr, 90))

    def _format_val(v: float) -> str:
        if "{}" in fmt or "{:" in fmt:
            return fmt.format(v)
        return f"{v:{fmt}}"

    med_lbl = f"Median: {_format_val(med)}"
    p10_lbl = f"p10: {_format_val(p10)}"
    p90_lbl = f"p90: {_format_val(p90)}"

    if is_horizontal:
        ax.axhline(med, color=MEDIAN, linestyle="-", linewidth=1.8, label=med_lbl, zorder=4)
        ax.axhline(p10, color=PERCENTILE, linestyle="--", linewidth=1.3, label=p10_lbl, zorder=4)
        ax.axhline(p90, color=PERCENTILE, linestyle=":", linewidth=1.3, label=p90_lbl, zorder=4)
    else:
        ax.axvline(med, color=MEDIAN, linestyle="-", linewidth=1.8, label=med_lbl, zorder=4)
        ax.axvline(p10, color=PERCENTILE, linestyle="--", linewidth=1.3, label=p10_lbl, zorder=4)
        ax.axvline(p90, color=PERCENTILE, linestyle=":", linewidth=1.3, label=p90_lbl, zorder=4)


def add_ref_line(
    ax: plt.Axes,
    y: float,
    label: str,
    is_vertical: bool = False,
    linestyle: str = "--",
    linewidth: float = 1.4,
) -> None:
    """Add a reference line in REF color (cam)."""
    if is_vertical:
        ax.axvline(y, color=REF, linestyle=linestyle, linewidth=linewidth, alpha=0.85, label=label, zorder=3)
    else:
        ax.axhline(y, color=REF, linestyle=linestyle, linewidth=linewidth, alpha=0.85, label=label, zorder=3)


def fmt_pct(ax: plt.Axes, axis: str = "y", decimals: int = 0) -> None:
    """Format axes ticks as percentages (e.g. 50%)."""
    formatter = ticker.FuncFormatter(lambda v, _: f"{v:.{decimals}f}%")
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(formatter)
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(formatter)


def fmt_money(ax: plt.Axes, axis: str = "x", decimals: int = 0) -> None:
    """Format axes ticks as currency (e.g. $50)."""
    formatter = ticker.FuncFormatter(lambda v, _: f"${v:,.{decimals}f}")
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(formatter)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(formatter)


def save_fig(fig: plt.Figure, part: str, name: str) -> Path:
    """Save matplotlib figure to reports/figures/<part>/<name>.png at 200 dpi."""
    file_name = name if name.endswith(".png") else f"{name}.png"
    out_dir = Path("reports") / "figures" / part
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / file_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return out_path


def plot_rate_and_excess(
    table: pd.DataFrame,
    title: str,
    order: list[str] | None = None,
) -> plt.Figure:
    """Single-panel rate and excess plot comparing observed return rate to R x F expectation.

    - Bar observed in BAR (#93C5FD), groups with Insight in PRIMARY (#1E40AF)
    - Expected R x F drawn as a horizontal dash in MEDIAN (#0F172A) on each bar
    - Reference horizontal line in REF (#F59E0B) for overall baseline rate
    - Annotate excess_pp on top of each bar
    - Subtitle / secondary label under each group name with household count
    - Callout box at upper left/right with summary verdict
    - Single panel FIGSIZE_FULL = (10, 5.6), zero dual y-axes
    """
    set_style()
    df = table.copy()

    # Identify the feature group column
    known_cols = {
        "n_rows",
        "n_households",
        "row_share",
        "observed",
        "expected",
        "excess_pp",
        "overall_rate",
        "raw_diff_pp",
        "same_sign_share",
        "n_cutoffs_used",
        "verdict",
        "verdict_reason",
    }
    candidate_cols = [c for c in df.columns if c not in known_cols]
    if not candidate_cols:
        raise ValueError("Không tìm thấy cột phân nhóm tính năng trong bảng")
    group_col = candidate_cols[0]

    # Reorder if specified
    if order is not None:
        df[group_col] = pd.Categorical(df[group_col].astype(str), categories=order, ordered=True)
        df = df.sort_values(group_col).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=FIGSIZE_FULL, dpi=150)

    n_groups = len(df)
    x = np.arange(n_groups)
    bar_width = 0.52

    observed_pct = df["observed"] * 100.0
    expected_pct = df["expected"] * 100.0
    overall_pct = float(df["overall_rate"].iloc[0]) * 100.0 if "overall_rate" in df.columns else 53.11

    # Bar colors:
    # - Insight with excess >= 0: PRIMARY (#1E40AF - xanh đậm)
    # - Insight with excess < 0: ACCENT (#DC2626 - đỏ cảnh báo)
    # - Non-insight: BAR (#93C5FD - xanh nhạt)
    bar_colors = []
    has_pos_insight = False
    has_neg_insight = False
    has_non_insight = False

    for _, row in df.iterrows():
        v = str(row.get("verdict", ""))
        ex = float(row.get("excess_pp", 0.0))
        if "Insight" in v:
            if ex >= 0:
                bar_colors.append(PRIMARY)
                has_pos_insight = True
            else:
                bar_colors.append(ACCENT)
                has_neg_insight = True
        else:
            bar_colors.append(BAR)
            has_non_insight = True

    bars = ax.bar(
        x,
        observed_pct,
        width=bar_width,
        color=bar_colors,
        edgecolor="white",
        linewidth=1.2,
        alpha=0.85,
        zorder=3,
    )

    # Expected R x F tick on each bar (short horizontal line)
    tick_half = bar_width * 0.42
    for i, exp_val in enumerate(expected_pct):
        ax.plot(
            [x[i] - tick_half, x[i] + tick_half],
            [exp_val, exp_val],
            color=MEDIAN,
            linewidth=2.8,
            solid_capstyle="round",
            zorder=5,
        )

    # Overall baseline horizontal line
    add_ref_line(ax, overall_pct, label=f"Rate chung ({overall_pct:.1f}%)")

    # Legend handles explaining bar types explicitly
    import matplotlib.patches as mpatches
    legend_handles = []
    if has_pos_insight:
        legend_handles.append(mpatches.Patch(color=PRIMARY, label="Insight (+ kỳ vọng)"))
    if has_neg_insight:
        legend_handles.append(mpatches.Patch(color=ACCENT, label="Insight (− kỳ vọng)"))
    if has_non_insight:
        legend_handles.append(mpatches.Patch(color=BAR, label="Không có thêm tín hiệu"))

    legend_handles.append(
        mpl.lines.Line2D([0], [0], color=MEDIAN, linewidth=2.8, solid_capstyle="round", label="Kỳ vọng R × F")
    )
    legend_handles.append(
        mpl.lines.Line2D([0], [0], color=REF, linestyle="--", linewidth=1.4, label=f"Rate chung ({overall_pct:.1f}%)")
    )

    # Annotate on top of bars: observed % and excess pp
    max_val = max(float(observed_pct.max()), float(expected_pct.max()))
    for i, bar in enumerate(bars):
        row = df.iloc[i]
        h = bar.get_height()
        exp_val = expected_pct.iloc[i]
        excess = row["excess_pp"]
        sign = "+" if excess > 0 else ""
        excess_str = f"{sign}{excess:.1f} pp"

        top_pos = max(h, exp_val) + 1.8
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            top_pos,
            f"{h:.1f}%\n({excess_str})",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=TITLE,
        )

    # Format xticklabels with group name and household count
    xtick_labels = []
    for _, row in df.iterrows():
        grp_name = str(row[group_col])
        if "n_households" in row and not pd.isna(row["n_households"]):
            n_hh = int(row["n_households"])
            xtick_labels.append(f"{grp_name}\n({n_hh:,} hộ)")
        else:
            xtick_labels.append(grp_name)

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=9.0)
    ax.set_xlabel(f"Phân tầng biến {group_col}", fontsize=9.5, fontweight="bold", labelpad=8)
    ax.set_ylabel("Tỷ lệ quay lại 7 ngày (%)", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, max(100.0, max_val + 16.0))
    fmt_pct(ax, axis="y")

    # Summary callout box (verdict tổng hợp)
    verdicts = df["verdict"].tolist() if "verdict" in df.columns else []
    n_total = len(verdicts)
    n_rf = sum(1 for v in verdicts if "R × F" in str(v))
    n_insight = sum(1 for v in verdicts if "Insight" in str(v))
    n_unsupported = sum(1 for v in verdicts if "Không được ủng hộ" in str(v))

    if n_rf == n_total:
        summary_txt = f"{n_total}/{n_total} nhóm: đã được R × F giải thích"
    elif n_insight > 0:
        summary_txt = f"{n_insight}/{n_total} nhóm Insight | {n_rf}/{n_total} nhóm R × F giải thích"
    elif n_unsupported > 0:
        summary_txt = f"{n_unsupported}/{n_total} nhóm không được ủng hộ | {n_rf}/{n_total} nhóm R × F giải thích"
    else:
        summary_txt = f"Tổng số {n_total} nhóm phân tích"

    annotate_box(ax, summary_txt, loc="upper right")

    ax.set_title(title, fontsize=12, fontweight="bold", color=TITLE, pad=12)
    ax.legend(handles=legend_handles, loc="upper left", frameon=True, facecolor=LEGEND_BG)

    fig.tight_layout()
    return fig

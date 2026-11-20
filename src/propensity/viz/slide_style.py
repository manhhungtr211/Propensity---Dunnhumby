"""Slide visualization style, color tokens, and helpers for presentation storytelling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Color tokens for slide presentations (Storytelling with Data: Red focal & Gray background)
RED: Final[str] = "#C8102E"        # Màu nhấn cốt lõi: điểm cần chú ý của slide
GRAY_BAR: Final[str] = "#CBD5E1"   # Series nền, bar phụ, vùng so sánh không quan trọng
GRAY_TEXT: Final[str] = "#64748B"  # Chữ phụ, nhãn trục, ghi chú
INK: Final[str] = "#1F2937"        # Chữ chính, đường dự đoán/kỳ vọng
BAND: Final[str] = "#FDE8EA"       # Dải tô nhạt (vùng ≤ 7 ngày, vùng ±3 pp)

# Deprecated aliases kept for backward compatibility
RED_LIGHT: Final[str] = "#F4A3A8"
NAVY: Final[str] = "#1E3A5F"

# Dimensions
SLIDE_FULL: Final[tuple[float, float]] = (13.0, 6.5)  # Ảnh chiếm cả slide
SLIDE_HALF: Final[tuple[float, float]] = (8.0, 6.0)   # Ảnh chiếm nửa slide (2 ảnh/slide)

SLIDES_DIR: Final[Path] = Path("reports/slides")


def set_slide_style() -> None:
    """Set global matplotlib style conforming to slide presentation standards.

    - Nền trắng
    - Font sans 13pt
    - Nhãn trục 13pt màu GRAY_TEXT
    - Số trên trục 12pt
    - Bỏ viền trên và phải
    - Grid trục y nét liền mảnh #E5E7EB
    - Bar viền trắng 1.5px
    - Legend không khung, đặt ở trên
    """
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "sans-serif"],
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.labelcolor": GRAY_TEXT,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.color": GRAY_TEXT,
        "ytick.color": GRAY_TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": "#E5E7EB",
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
        "patch.edgecolor": "white",
        "patch.linewidth": 1.5,
        "legend.frameon": False,
        "legend.fontsize": 12,
        "legend.loc": "upper center",
        "figure.dpi": 200,
    })


def annotate_point(
    ax: plt.Axes,
    xy: tuple[float, float],
    text: str,
    xytext: tuple[float, float],
    *,
    color: str = INK,
    fontsize: int = 13,
    fontweight: str = "bold",
    ha: str = "center",
    va: str = "bottom",
    **kwargs: Any,
) -> Any:
    """Annotate a specific data point with a thin INK arrow and bold text."""
    arrowprops = dict(
        arrowstyle="->",
        color=INK,
        lw=1.2,
        shrinkA=3,
        shrinkB=3,
    )
    return ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        textcoords="data",
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        arrowprops=arrowprops,
        **kwargs,
    )


def fmt_pct(val: float, decimals: int = 1, *, is_ratio: bool = False) -> str:
    """Format a percentage value with Vietnamese decimal comma (e.g. 53,1%)."""
    num = val * 100.0 if is_ratio else val
    return f"{num:.{decimals}f}%".replace(".", ",")


def fmt_num(val: float | int, decimals: int = 0) -> str:
    """Format a number with Vietnamese thousands dot and decimal comma (e.g. 2.500, 1.322,4)."""
    if decimals == 0:
        return f"{int(round(val)):,}".replace(",", ".")
    base = f"{val:,.{decimals}f}"
    return base.replace(",", "_").replace(".", ",").replace("_", ".")


def fmt_money(val: float, decimals: int = 1) -> str:
    """Format currency in dollars with Vietnamese decimal comma (e.g. $3,8)."""
    return f"${val:.{decimals}f}".replace(".", ",")


def pct_axis(
    ax: plt.Axes,
    axis: str = "y",
    xmax: float | None = None,
    decimals: int = 0,
) -> None:
    """Format an axis to display percentage values with Vietnamese comma decimals."""
    target_axis = ax.xaxis if axis == "x" else ax.yaxis
    if xmax is None:
        lim = ax.get_xlim() if axis == "x" else ax.get_ylim()
        xmax = 1.0 if max(abs(lim[0]), abs(lim[1])) <= 1.05 else 100.0

    def _formatter(x: float, pos: int | None = None) -> str:
        val = (x / xmax) * 100.0
        if decimals == 0:
            return f"{int(round(val))}%"
        return f"{val:.{decimals}f}%".replace(".", ",")

    target_axis.set_major_formatter(ticker.FuncFormatter(_formatter))


def thousands_axis(ax: plt.Axes, axis: str = "y") -> None:
    """Format an axis with Vietnamese thousands dots (e.g. 1.200, 2.500)."""
    target_axis = ax.xaxis if axis == "x" else ax.yaxis
    target_axis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{int(round(x)):,}".replace(",", "."))
    )



def hide_value_axis(ax: plt.Axes, axis: str = "y") -> None:
    """Hide the value axis and its grid lines completely, leaving only the zero baseline."""
    if axis == "y":
        ax.yaxis.grid(False)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel("")
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
    elif axis == "x":
        ax.xaxis.grid(False)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel("")
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(True)


def direct_label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str = INK,
    fontsize: int = 11,
    fontweight: str = "normal",
    ha: str = "center",
    va: str = "bottom",
    **kwargs: Any,
) -> Any:
    """Directly place a label near a data point or bar without needing a legend."""
    return ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        zorder=5,
        **kwargs,
    )


def save_slide(fig: plt.Figure, name: str) -> tuple[Path, Path]:
    """Save a slide figure as PNG (200 dpi) and SVG in reports/slides/."""
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = SLIDES_DIR / f"{name}.png"
    svg_path = SLIDES_DIR / f"{name}.svg"

    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    return png_path, svg_path

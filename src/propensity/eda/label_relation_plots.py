"""Visualization functions for R x F baseline, excess over expectation, and feature verdicts."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_rf_heatmap(cells: pd.DataFrame, overall_rate: float) -> plt.Figure:
    """Plot canonical R x F baseline return rate heatmap with sample size annotations.

    Rows: Recency bins (e.g. 1–7, 8–14, 15–30, 31+).
    Cols: Frequency bins (5 numeric intervals + short history).
    Unreliable cells (n_households < 50) are shaded with hatched patterns.
    """
    # Determine canonical orders
    if "r_bin" in cells.columns:
        r_col = "r_bin"
    elif "recency_bin" in cells.columns:
        r_col = "recency_bin"
    else:
        raise ValueError("cells phải có cột 'r_bin' hoặc 'recency_bin'")

    if "f_bin" in cells.columns:
        f_col = "f_bin"
    elif "F_bin" in cells.columns:
        f_col = "F_bin"
    else:
        raise ValueError("cells phải có cột 'f_bin' hoặc 'F_bin'")

    # Deduce row and col order
    canonical_r = ["1–7", "8–14", "15–30", "31+"]
    unique_r = list(cells[r_col].unique())
    r_order = [r for r in canonical_r if r in unique_r] + [r for r in unique_r if r not in canonical_r]

    # Frequency order: preserve categorical order if categorical, else sort
    if isinstance(cells[f_col].dtype, pd.CategoricalDtype):
        f_order = [f for f in cells[f_col].cat.categories if f in cells[f_col].values]
    else:
        # Put "Lịch sử ngắn" at the end if present
        unique_f = list(cells[f_col].unique())
        short_hist = [f for f in unique_f if "ngắn" in str(f).lower() or "<28" in str(f)]
        regular_f = [f for f in unique_f if f not in short_hist]
        f_order = regular_f + short_hist

    # Pivot rate and metadata matrices
    rate_matrix = pd.pivot_table(
        cells, index=r_col, columns=f_col, values="rate", aggfunc="mean"
    ).reindex(index=r_order, columns=f_order) * 100.0

    hh_matrix = pd.pivot_table(
        cells, index=r_col, columns=f_col, values="n_households", aggfunc="first"
    ).reindex(index=r_order, columns=f_order)

    reliable_matrix = pd.pivot_table(
        cells, index=r_col, columns=f_col, values="is_reliable", aggfunc="first"
    ).reindex(index=r_order, columns=f_order).fillna(False)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    # Diverging colormap centered at overall_rate * 100
    vcenter = overall_rate * 100.0
    vmin = max(0.0, float(rate_matrix.min().min()) if not np.isnan(rate_matrix.min().min()) else 0.0)
    vmax = min(100.0, float(rate_matrix.max().max()) if not np.isnan(rate_matrix.max().max()) else 100.0)
    if vmin >= vcenter:
        vmin = max(0.0, vcenter - 20.0)
    if vmax <= vcenter:
        vmax = min(100.0, vcenter + 20.0)

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap = plt.colormaps["RdYlBu_r"].copy()

    im = ax.imshow(rate_matrix.values, cmap=cmap, norm=norm, aspect="auto")

    # Set ticks and labels
    ax.set_xticks(range(len(f_order)))
    ax.set_xticklabels(f_order, fontsize=9.5, fontweight="semibold")
    ax.set_xlabel("Cường độ mua sắm F_c (lần mua/tuần, DAY < c)", fontsize=11, fontweight="bold", labelpad=8)

    ax.set_yticks(range(len(r_order)))
    ax.set_yticklabels(r_order, fontsize=10, fontweight="semibold")
    ax.set_ylabel("Tầng Recency (ngày kể từ lần mua gần nhất)", fontsize=11, fontweight="bold", labelpad=8)

    ax.set_title(
        f"Lưới cơ sở Recency × Frequency (Baseline R × F)\n"
        f"[Tâm chuẩn hóa tại baseline chung = {vcenter:.1f}% | Ô gạch xám: Ít dữ liệu (<50 hộ)]",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    # Cell annotations
    for i in range(len(r_order)):
        for j in range(len(f_order)):
            val = rate_matrix.iloc[i, j]
            n_hh = hh_matrix.iloc[i, j]
            is_rel = reliable_matrix.iloc[i, j]

            if pd.isna(val):
                # Empty cell
                rect = mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=True, facecolor="#F3F4F6", edgecolor="#D1D5DB"
                )
                ax.add_patch(rect)
                ax.text(j, i, "Không có DL", ha="center", va="center", fontsize=8.5, color="#9CA3AF")
                continue

            if not is_rel:
                # Unreliable cell: hatch pattern
                rect = mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=True, facecolor="#E5E7EB", hatch="//", edgecolor="#9CA3AF", alpha=0.9
                )
                ax.add_patch(rect)
                ax.text(
                    j,
                    i,
                    f"{val:.1f}%\n(Ít DL: {int(n_hh)} hộ)",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="#4B5563",
                    fontweight="bold",
                )
            else:
                # High contrast text depending on background lightness
                text_color = "white" if (val > vcenter + 18.0 or val < vcenter - 22.0) else "#111827"
                ax.text(
                    j,
                    i,
                    f"{val:.1f}%\n({int(n_hh):,} hộ)",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                    fontweight="bold",
                )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label("Tỷ lệ quay lại trong 7 ngày (%)", fontsize=10, fontweight="bold")
    cbar.ax.axhline(vcenter, color="black", linestyle="--", linewidth=1.5)
    cbar.ax.text(
        1.15,
        vcenter,
        f" Chung: {vcenter:.1f}%",
        va="center",
        ha="left",
        transform=cbar.ax.get_yaxis_transform(),
        fontsize=8.5,
        fontweight="bold",
        color="#1F2937",
    )

    return fig


def plot_rate_and_excess(
    table: pd.DataFrame,
    title: str,
    order: list[str] | None = None,
) -> plt.Figure:
    """Plot 2-panel chart: Upper panel shows observed return rate, expected rate, excess and verdict; Lower panel shows sample share.

    No error bars used; clean, high-contrast visual design.
    """
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

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(11, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]},
        constrained_layout=True,
    )

    x = np.arange(len(df))
    bar_width = 0.52

    # Upper Panel: Rates & Expectations
    observed_pct = df["observed"] * 100.0
    expected_pct = df["expected"] * 100.0
    overall_pct = float(df["overall_rate"].iloc[0]) * 100.0

    bars = ax_top.bar(
        x,
        observed_pct,
        width=bar_width,
        color="#2563EB",
        edgecolor="#1D4ED8",
        alpha=0.88,
        label="Tỷ lệ thực tế (Observed)",
        zorder=3,
    )

    # Expected point as diamond scatter
    ax_top.scatter(
        x,
        expected_pct,
        color="#EF4444",
        s=80,
        marker="D",
        edgecolors="#991B1B",
        linewidths=1.2,
        label="Kỳ vọng theo R × F (Expected)",
        zorder=5,
    )

    # Overall baseline horizontal line
    ax_top.axhline(
        overall_pct,
        color="#4B5563",
        linestyle="--",
        linewidth=1.4,
        label=f"Baseline chung ({overall_pct:.1f}%)",
        zorder=2,
    )

    # Annotate top of bars
    for i, bar in enumerate(bars):
        row = df.iloc[i]
        h = bar.get_height()
        excess = row["excess_pp"]
        verdict = row.get("verdict", "")
        sign = "+" if excess > 0 else ""
        excess_str = f"{sign}{excess:.1f} pp"

        label_lines = [f"{h:.1f}%", f"Δ: {excess_str}"]
        if verdict:
            # Shorten label for neat display
            short_verdict = verdict.replace("Đã được R × F giải thích", "R×F giải thích")
            label_lines.append(f"[{short_verdict}]")

        ax_top.text(
            bar.get_x() + bar.get_width() / 2.0,
            max(h, expected_pct.iloc[i]) + 2.0,
            "\n".join(label_lines),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#1F2937",
        )

    ax_top.set_ylabel("Tỷ lệ quay lại 7 ngày (%)", fontsize=10.5, fontweight="bold")
    ax_top.set_ylim(0, max(100.0, float(max(observed_pct.max(), expected_pct.max())) + 15.0))
    ax_top.grid(axis="y", linestyle=":", alpha=0.5, zorder=1)
    ax_top.legend(frameon=True, facecolor="white", edgecolor="#E5E7EB", loc="upper left", fontsize=9)
    ax_top.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # Lower Panel: Row Share
    share_pct = df["row_share"] * 100.0
    share_bars = ax_bottom.bar(
        x,
        share_pct,
        width=bar_width,
        color="#94A3B8",
        edgecolor="#64748B",
        alpha=0.85,
        zorder=3,
    )

    for bar, (_, row) in zip(share_bars, df.iterrows()):
        h = bar.get_height()
        ax_bottom.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.5,
            f"{h:.1f}%\n({int(row['n_households']):,} hộ)",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#334155",
            fontweight="semibold",
        )

    ax_bottom.set_ylabel("Tỷ trọng (%)", fontsize=10, fontweight="bold")
    ax_bottom.set_ylim(0, max(30.0, float(share_pct.max()) * 1.35))
    ax_bottom.grid(axis="y", linestyle=":", alpha=0.5, zorder=1)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(df[group_col].astype(str), fontsize=9.5, fontweight="semibold")
    ax_bottom.set_xlabel(f"Phân tầng biến {group_col}", fontsize=10.5, fontweight="bold", labelpad=6)

    return fig

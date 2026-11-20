"""Core analytical rate calculations, R x F baseline, excess over expectation, and verdict classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FindingThresholds:
    """Thresholds for classifying feature findings against R x F baseline."""

    min_effect_pp: float = 3.0        # Chênh tối thiểu so với kỳ vọng (pp)
    min_same_sign: float = 0.80       # Cùng dấu ở ≥ 80% cutoff
    min_row_share: float = 0.02       # Nhóm chiếm ≥ 2% số dòng
    min_households: int = 50          # Nhóm có ≥ 50 hộ
    min_rows_per_cutoff: int = 20     # Cutoff có nhóm dưới 20 dòng thì không tính vào same_sign


@dataclass(frozen=True)
class RFBaseline:
    """Canonical R x F baseline container."""

    cells: pd.DataFrame    # r_bin, f_bin, n_rows, n_households, n_pos, rate, row_share, is_reliable
    panel: pd.DataFrame    # panel có thêm expected_rf, expected_source
    overall_rate: float


def asof_aggregate(
    table: pd.DataFrame,
    cutoffs: list[int] | tuple[int, ...] | np.ndarray,
    agg_fn: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    day_col: str = "DAY",
    key: str = "household_key",
) -> pd.DataFrame:
    """Calculate point-in-time features strictly as-of each cutoff (DAY < c).

    Ghi chú: Nên gộp table về household × DAY trước khi gọi để tối ưu hiệu năng.
    """
    if day_col not in table.columns:
        raise ValueError(f"Thiếu cột thời gian '{day_col}' trong dữ liệu")
    if key not in table.columns:
        raise ValueError(f"Thiếu cột định danh '{key}' trong dữ liệu")

    frames: list[pd.DataFrame] = []
    for c in cutoffs:
        sub = table[table[day_col] < c]
        res = agg_fn(sub)
        if not isinstance(res, pd.DataFrame):
            raise ValueError("agg_fn phải trả về pandas DataFrame")
        if key not in res.columns:
            raise ValueError(f"Kết quả của agg_fn thiếu cột key '{key}'")
        if res[key].duplicated().any():
            raise ValueError(f"Cột key '{key}' bị trùng trong kết quả agg_fn tại cutoff {c}")

        res = res.copy()
        res["cutoff_day"] = c
        frames.append(res)

    if not frames:
        return pd.DataFrame(columns=[key, "cutoff_day"])

    return pd.concat(frames, ignore_index=True)


def join_features(panel: pd.DataFrame, *frames: pd.DataFrame) -> pd.DataFrame:
    """Left-join feature frames to panel on (household_key, cutoff_day).

    Raises ValueError if:
    - Any frame has duplicate keys (household_key, cutoff_day).
    - Any feature column in frame already exists in panel (column collision).
    - Resulting row count differs from initial panel row count.
    """
    keys = ["household_key", "cutoff_day"]
    for k in keys:
        if k not in panel.columns:
            raise ValueError(f"Panel thiếu cột khóa '{k}'")

    current_panel = panel.copy()
    initial_len = len(current_panel)

    for i, frame in enumerate(frames):
        for k in keys:
            if k not in frame.columns:
                raise ValueError(f"Frame #{i} thiếu cột khóa '{k}'")

        if frame.duplicated(subset=keys).any():
            raise ValueError(f"Frame #{i} chứa khóa trùng lặp trên {keys}")

        non_key_cols = [c for c in frame.columns if c not in keys]
        collision = set(non_key_cols) & set(current_panel.columns)
        if collision:
            raise ValueError(f"Xung đột tên cột bị đè giữa frame #{i} và panel: {collision}")

        current_panel = current_panel.merge(frame, on=keys, how="left")
        if len(current_panel) != initial_len:
            raise ValueError(f"Số dòng sau khi join frame #{i} ({len(current_panel)}) khác số dòng ban đầu ({initial_len})")

    return current_panel


def rf_baseline(
    panel: pd.DataFrame,
    r_col: str = "recency_bin",
    f_col: str = "F_bin",
    label_col: str = "target",
    min_households: int = 50,
) -> RFBaseline:
    """Construct R x F baseline table and map expected return rates to panel.

    For reliable cells (n_households >= min_households):
        expected_rf = pooled rate of the cell across all cutoffs.
        expected_source = 'cell'
    For unreliable cells:
        expected_rf = pooled rate of the entire recency row (r_margin).
        expected_source = 'r_margin'
    """
    req_cols = [r_col, f_col, label_col, "household_key"]
    missing = [c for c in req_cols if c not in panel.columns]
    if missing:
        raise ValueError(f"Panel thiếu các cột: {missing}")

    overall_rate = float(panel[label_col].mean())

    # Calculate cells across R x F
    grouped = (
        panel.groupby([r_col, f_col], observed=False)
        .agg(
            n_rows=(label_col, "count"),
            n_households=("household_key", "nunique"),
            n_pos=(label_col, "sum"),
            rate=(label_col, "mean"),
        )
        .reset_index()
    )

    # Exclude empty cells
    cells = grouped[grouped["n_rows"] > 0].copy()
    cells["row_share"] = cells["n_rows"] / len(panel)
    cells["is_reliable"] = cells["n_households"] >= min_households

    # Rename to standardized columns
    cells = cells.rename(columns={r_col: "r_bin", f_col: "f_bin"})

    # Recency margin rates for fallback
    r_margin_map = panel.groupby(r_col, observed=False)[label_col].mean().to_dict()

    # Reliable cell rate lookup
    reliable_cells = cells[cells["is_reliable"]]
    cell_rate_map = dict(
        zip(zip(reliable_cells["r_bin"], reliable_cells["f_bin"]), reliable_cells["rate"])
    )

    # Vectorized / looped mapping to panel
    expected_rf_vals: list[float] = []
    expected_source_vals: list[str] = []

    for r_val, f_val in zip(panel[r_col], panel[f_col]):
        key = (r_val, f_val)
        if key in cell_rate_map:
            expected_rf_vals.append(cell_rate_map[key])
            expected_source_vals.append("cell")
        else:
            expected_rf_vals.append(r_margin_map.get(r_val, overall_rate))
            expected_source_vals.append("r_margin")

    panel_out = panel.copy()
    panel_out["expected_rf"] = expected_rf_vals
    panel_out["expected_source"] = expected_source_vals

    return RFBaseline(
        cells=cells.reset_index(drop=True),
        panel=panel_out,
        overall_rate=overall_rate,
    )


def excess_over_rf(
    panel: pd.DataFrame,
    group_col: str,
    label_col: str = "target",
    expected_col: str = "expected_rf",
    thr: FindingThresholds | None = None,
) -> pd.DataFrame:
    """Calculate group return rate and excess over R x F baseline.

    Missing values in group_col are mapped to string 'null'.
    """
    if thr is None:
        thr = FindingThresholds()

    req_cols = [group_col, label_col, expected_col, "household_key", "cutoff_day"]
    missing = [c for c in req_cols if c not in panel.columns]
    if missing:
        raise ValueError(f"Thiếu các cột bắt buộc: {missing}")

    # Process group column with explicit 'null' handling
    s = panel[group_col].copy()
    if isinstance(s.dtype, pd.CategoricalDtype):
        if "null" not in s.cat.categories:
            s = s.cat.add_categories(["null"])
        s = s.fillna("null")
    else:
        s = s.fillna("null").astype(str)

    temp_panel = panel[[label_col, expected_col, "household_key", "cutoff_day"]].copy()
    temp_panel["__group__"] = s

    overall_rate = float(temp_panel[label_col].mean())
    total_rows = len(temp_panel)
    all_cutoffs = sorted(temp_panel["cutoff_day"].unique())

    records: list[dict[str, Any]] = []

    # Iterate over unique groups preserving categorical order if applicable
    if isinstance(s.dtype, pd.CategoricalDtype):
        categories = [cat for cat in s.cat.categories if (s == cat).any()]
    else:
        categories = list(s.unique())

    for g in categories:
        grp = temp_panel[temp_panel["__group__"] == g]
        n_rows = len(grp)
        if n_rows == 0:
            continue

        n_households = int(grp["household_key"].nunique())
        row_share = float(n_rows / total_rows)
        observed = float(grp[label_col].mean())
        expected = float(grp[expected_col].mean())
        excess_pp = float((observed - expected) * 100.0)
        raw_diff_pp = float((observed - overall_rate) * 100.0)

        # Same sign share calculation across cutoffs
        n_cutoffs_used = 0
        n_same_sign = 0

        for c in all_cutoffs:
            sub = grp[grp["cutoff_day"] == c]
            if len(sub) >= thr.min_rows_per_cutoff:
                n_cutoffs_used += 1
                diff_c = float(sub[label_col].mean() - sub[expected_col].mean())
                if excess_pp == 0.0:
                    n_same_sign += 1
                elif excess_pp > 0.0 and diff_c > 0.0:
                    n_same_sign += 1
                elif excess_pp < 0.0 and diff_c < 0.0:
                    n_same_sign += 1

        same_sign_share = float(n_same_sign / n_cutoffs_used) if n_cutoffs_used > 0 else 0.0

        records.append(
            {
                group_col: g,
                "n_rows": n_rows,
                "n_households": n_households,
                "row_share": row_share,
                "observed": observed,
                "expected": expected,
                "excess_pp": excess_pp,
                "overall_rate": overall_rate,
                "raw_diff_pp": raw_diff_pp,
                "same_sign_share": same_sign_share,
                "n_cutoffs_used": n_cutoffs_used,
            }
        )

    res_df = pd.DataFrame(records)
    return res_df


def classify_verdict(
    row: pd.Series | dict[str, Any],
    thr: FindingThresholds | None = None,
) -> tuple[str, str]:
    """Classify a feature's effect into a structured verdict and rationale.

    Check order:
    1. Sample size: row_share < min_row_share OR n_households < min_households -> 'Không đủ dữ liệu'
    2. R x F explanation: |raw_diff_pp| >= min_effect_pp AND |excess_pp| < min_effect_pp -> 'Đã được R × F giải thích'
    3. Unsupported: |excess_pp| < min_effect_pp -> 'Không được ủng hộ'
    4. Stability: same_sign_share < min_same_sign -> 'Không ổn định'
    5. Supported Insight: else -> 'Insight'
    """
    if thr is None:
        thr = FindingThresholds()

    row_share = float(row["row_share"])
    n_households = int(row["n_households"])
    raw_diff_pp = float(row["raw_diff_pp"])
    excess_pp = float(row["excess_pp"])
    same_sign_share = float(row["same_sign_share"])

    # 1. Sample size
    if row_share < thr.min_row_share or n_households < thr.min_households:
        return (
            "Không đủ dữ liệu",
            f"Nhóm quá nhỏ ({row_share:.1%} dòng < {thr.min_row_share:.0%} hoặc {n_households} hộ < {thr.min_households})",
        )

    # 2. Fully explained by R x F
    if abs(raw_diff_pp) >= thr.min_effect_pp and abs(excess_pp) < thr.min_effect_pp:
        return (
            "Đã được R × F giải thích",
            f"Chênh lệch thô {raw_diff_pp:+.2f} pp nhưng sau R × F chỉ còn {excess_pp:+.2f} pp (< {thr.min_effect_pp:.1f} pp)",
        )

    # 3. Unsupported by evidence
    if abs(excess_pp) < thr.min_effect_pp:
        return (
            "Không được ủng hộ",
            f"Chênh lệch sau R × F ({excess_pp:+.2f} pp) nhỏ hơn ngưỡng tối thiểu ({thr.min_effect_pp:.1f} pp)",
        )

    # 4. Unstable across temporal cutoffs
    if same_sign_share < thr.min_same_sign:
        return (
            "Không ổn định",
            f"Cùng dấu ở {same_sign_share:.1%} cutoff (< {thr.min_same_sign:.0%})",
        )

    # 5. True insight
    return (
        "Insight",
        f"Hiệu ứng {excess_pp:+.2f} pp ổn định qua {same_sign_share:.1%} cutoff",
    )


def add_verdicts(
    table: pd.DataFrame,
    thr: FindingThresholds | None = None,
) -> pd.DataFrame:
    """Add verdict and verdict_reason columns to excess_over_rf table."""
    if thr is None:
        thr = FindingThresholds()

    verdicts: list[str] = []
    reasons: list[str] = []

    for _, row in table.iterrows():
        v, r = classify_verdict(row, thr)
        verdicts.append(v)
        reasons.append(r)

    out = table.copy()
    out["verdict"] = verdicts
    out["verdict_reason"] = reasons
    return out


def quantile_bin(
    s: pd.Series,
    q: int = 5,
    null_label: str = "null",
    prefix: str = "Q",
) -> pd.Categorical:
    """Discretize continuous series into quantiles with numerical interval labels and explicit null group.

    Uses pd.qcut(..., duplicates='drop'). When duplicate values exist, fewer than q bins may be produced.
    Null values are mapped to null_label and placed as the last category.
    """
    valid = s.dropna()
    if valid.empty:
        return pd.Categorical([null_label] * len(s), categories=[null_label], ordered=True)

    if valid.nunique() == 1:
        val = valid.iloc[0]
        label = f"{prefix}1 (={val:.2f})"
        categories = [label]
        if s.isna().any():
            categories.append(null_label)
        res = s.map({val: label}).fillna(null_label)
        return pd.Categorical(res, categories=categories, ordered=True)

    _, bins = pd.qcut(valid, q=q, retbins=True, duplicates="drop")
    k = len(bins) - 1
    if k <= 1:
        labels = [f"{prefix}1"]
    else:
        labels = []
        for i in range(k):
            if i == 0:
                labels.append(f"{prefix}{i+1} (≤{bins[i+1]:.2f})")
            elif i == k - 1:
                labels.append(f"{prefix}{i+1} (>{bins[i]:.2f})")
            else:
                labels.append(f"{prefix}{i+1} ({bins[i]:.2f}–{bins[i+1]:.2f})")

    binned = pd.qcut(valid, q=q, labels=labels, duplicates="drop")
    categories = list(labels)
    if s.isna().any():
        categories.append(null_label)

    res = pd.Series(index=s.index, dtype="object")
    res.loc[valid.index] = binned.astype(str)
    res = res.fillna(null_label)
    return pd.Categorical(res, categories=categories, ordered=True)


def classify_finding(
    metric_df: pd.DataFrame,
    strata_matrix: pd.DataFrame | None = None,
    thresholds: Any = None,
) -> dict[str, Any]:
    """DEPRECATED: Dùng add_verdicts() hoặc classify_verdict() thay thế."""
    from propensity.eda.customer_label_relation import (
        classify_finding as _orig_classify_finding,
    )
    return _orig_classify_finding(metric_df, strata_matrix, thresholds)


def feature_signal_summary(
    results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize raw vs R x F adjusted signal for candidate features.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        Dictionary mapping feature display name to its excess_over_rf table with verdicts.

    Returns
    -------
    pd.DataFrame
        Columns: feature, raw_pp, adj_pp, best_verdict.
        Sorted descending by raw_pp.
    """
    records: list[dict[str, Any]] = []

    for name, df in results.items():
        if df is None or df.empty:
            continue

        group_col = df.columns[0]
        sub = df.copy()

        # Bỏ bin "Không đủ dữ liệu"
        if "verdict" in sub.columns:
            sub = sub[sub["verdict"] != "Không đủ dữ liệu"]

        # Bỏ bin null
        group_vals = sub[group_col].astype(str).str.strip().str.lower()
        sub = sub[~group_vals.isin(["null", "nan", "none", ""])]

        if sub.empty:
            continue

        raw_pp = float(sub["raw_diff_pp"].abs().max()) if "raw_diff_pp" in sub.columns else 0.0
        adj_pp = float(sub["excess_pp"].abs().max()) if "excess_pp" in sub.columns else 0.0

        verdicts = sub["verdict"].tolist() if "verdict" in sub.columns else []
        if "Insight" in verdicts:
            best_verdict = "Insight"
        elif "Đã được R × F giải thích" in verdicts:
            best_verdict = "Đã được R × F giải thích"
        elif "Không được ủng hộ" in verdicts:
            best_verdict = "Không được ủng hộ"
        elif "Không ổn định" in verdicts:
            best_verdict = "Không ổn định"
        elif verdicts:
            best_verdict = verdicts[0]
        else:
            best_verdict = "Không xác định"

        records.append({
            "feature": name,
            "raw_pp": raw_pp,
            "adj_pp": adj_pp,
            "best_verdict": best_verdict,
        })

    summary_df = pd.DataFrame(records)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("raw_pp", ascending=False).reset_index(drop=True)
    else:
        summary_df = pd.DataFrame(columns=["feature", "raw_pp", "adj_pp", "best_verdict"])
    return summary_df


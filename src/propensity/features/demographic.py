"""Demographic feature grouping, panel enrichment, coverage profiling, and excess analysis."""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd

from propensity.eda.propensity_rates import FindingThresholds, add_verdicts, excess_over_rf

DEMO_GROUPS: Mapping[str, Mapping[str, Sequence[str]]] = {
    "age": {
        "19-34": ["19-24", "25-34"],
        "35-44": ["35-44"],
        "45-54": ["45-54"],
        "55+": ["55-64", "65+"],
    },
    "income": {
        "<35K": ["Under 15K", "15-24K", "25-34K"],
        "35-49K": ["35-49K"],
        "50-74K": ["50-74K"],
        "75-124K": ["75-99K", "100-124K"],
        "125K+": ["125-149K", "150-174K", "175-199K", "200-249K", "250K+"],
    },
    "homeowner": {
        "Sở hữu": ["Homeowner", "Probable Owner"],
        "Thuê": ["Renter", "Probable Renter"],
        "Không rõ": ["Unknown"],
    },
    "marital": {
        "A": ["A"],  # Married per Dunnhumby specs
        "B": ["B"],  # Single per Dunnhumby specs
        "U": ["U"],  # Unknown
    },
    "hh_comp": {
        "Độc thân": ["Single Female", "Single Male"],
        "2 người lớn, không con": ["2 Adults No Kids"],
        "Có con": ["1 Adult Kids", "2 Adults Kids"],
        "Không rõ": ["Unknown"],
    },
    "hh_size": {
        "1": ["1"],
        "2": ["2"],
        "3": ["3"],
        "4+": ["4", "5+"],
    },
    "kids": {
        "Không/không rõ": ["None/Unknown"],
        "1": ["1"],
        "2+": ["2", "3+"],
    },
}

RAW_MAP: Mapping[str, str] = {
    "age": "AGE_DESC",
    "income": "INCOME_DESC",
    "homeowner": "HOMEOWNER_DESC",
    "marital": "MARITAL_STATUS_CODE",
    "hh_comp": "HH_COMP_DESC",
    "hh_size": "HOUSEHOLD_SIZE_DESC",
    "kids": "KID_CATEGORY_DESC",
}


def group_demographic(
    demo: pd.DataFrame,
    groups: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> pd.DataFrame:
    """Map raw demographic categories to coarsened groups (each >= 50 households).

    Parameters
    ----------
    demo : pd.DataFrame
        Raw household demographic table containing household_key and descriptor columns.
    groups : Mapping, optional
        Target group mapping. Defaults to DEMO_GROUPS.

    Returns
    -------
    pd.DataFrame
        One row per household with columns: household_key, and 7 demo_<attr> ordered categoricals.
    """
    if "household_key" not in demo.columns:
        raise ValueError("Bảng demo thiếu cột 'household_key'")

    if groups is None:
        groups = DEMO_GROUPS

    df_out = demo[["household_key"]].copy()

    for attr, mapping in groups.items():
        raw_col = RAW_MAP.get(attr, attr)
        if raw_col not in demo.columns and attr in demo.columns:
            raw_col = attr
        if raw_col not in demo.columns:
            raise ValueError(f"Không tìm thấy cột tương ứng cho thuộc tính '{attr}' trong demo")

        # Build reverse map
        inv_map: dict[str, str] = {}
        for target_level, raw_vals in mapping.items():
            for r in raw_vals:
                inv_map[str(r).strip()] = target_level

        # Validate unmapped values
        unique_raw = [str(x).strip() for x in demo[raw_col].dropna().unique()]
        unmapped = set(unique_raw) - set(inv_map.keys())
        if unmapped:
            raise ValueError(f"Có giá trị không map được trong {raw_col}: {unmapped}")

        clean_series = demo[raw_col].astype(str).str.strip().map(inv_map)
        categories = list(mapping.keys())
        df_out[f"demo_{attr}"] = pd.Categorical(clean_series, categories=categories, ordered=True)

    return df_out.sort_values("household_key").reset_index(drop=True)


def demographic_features(
    panel: pd.DataFrame,
    demo_grouped: pd.DataFrame,
) -> pd.DataFrame:
    """Merge static demographic features into the cutoff panel.

    Households without demographic record receive has_demographic = 0 and
    demo_<attr> = 'Không có dữ liệu'.

    Parameters
    ----------
    panel : pd.DataFrame
        Customer cutoff panel.
    demo_grouped : pd.DataFrame
        Output of group_demographic.

    Returns
    -------
    pd.DataFrame
        Panel with has_demographic and 7 demo_<attr> columns added.
    """
    if "household_key" not in panel.columns:
        raise ValueError("Bảng panel thiếu cột 'household_key'")
    if "household_key" not in demo_grouped.columns:
        raise ValueError("Bảng demo_grouped thiếu cột 'household_key'")

    res = panel.copy()
    demo_cols = [c for c in demo_grouped.columns if c != "household_key"]

    merged = res.merge(demo_grouped, on="household_key", how="left")
    merged["has_demographic"] = np.where(merged["demo_age"].notna(), 1, 0)

    for col in demo_cols:
        orig_cats = list(demo_grouped[col].cat.categories)
        new_cats = orig_cats + ["Không có dữ liệu"]
        merged[col] = pd.Categorical(
            merged[col].astype(object).fillna("Không có dữ liệu"),
            categories=new_cats,
            ordered=True,
        )

    return merged


def demographic_coverage_profile(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute behavioral profile and distribution across F and Recency by has_demographic.

    Parameters
    ----------
    panel : pd.DataFrame
        Cutoff panel enriched with has_demographic, F_bin, and recency_bin.

    Returns
    -------
    pd.DataFrame
        Summary table comparing households with vs without demographic.
    """
    req_cols = {"has_demographic", "household_key", "target", "expected_rf"}
    missing = req_cols - set(panel.columns)
    if missing:
        raise ValueError(f"Bảng panel thiếu các cột: {missing}")

    f_col = "F_bin" if "F_bin" in panel.columns else "f_bin"
    r_col = "recency_bin" if "recency_bin" in panel.columns else "r_bin"

    total_rows = len(panel)
    records = []

    for has_d in [1, 0]:
        sub = panel[panel["has_demographic"] == has_d]
        n_hh = int(sub["household_key"].nunique())
        n_rows = len(sub)
        row_share = n_rows / total_rows if total_rows > 0 else 0.0
        obs = float(sub["target"].mean())
        exp = float(sub["expected_rf"].mean())
        excess = (obs - exp) * 100.0

        label = "Có demographic" if has_d == 1 else "Không có demographic"

        rec: dict[str, Any] = {
            "group": label,
            "has_demographic": has_d,
            "n_households": n_hh,
            "n_rows": n_rows,
            "row_share": row_share,
            "observed": obs,
            "expected": exp,
            "excess_pp": excess,
        }

        # Add F_bin distributions
        if f_col in panel.columns:
            f_counts = sub[f_col].value_counts(normalize=True)
            for f_val in panel[f_col].cat.categories:
                rec[f"f_{f_val}"] = float(f_counts.get(f_val, 0.0))

        # Add recency_bin distributions
        if r_col in panel.columns:
            r_counts = sub[r_col].value_counts(normalize=True)
            for r_val in panel[r_col].cat.categories:
                rec[f"r_{r_val}"] = float(r_counts.get(r_val, 0.0))

        records.append(rec)

    return pd.DataFrame(records)


def attribute_excess_table(
    panel: pd.DataFrame,
    attrs: Sequence[str] | None = None,
    thr: FindingThresholds | None = None,
) -> pd.DataFrame:
    """Calculate excess over R x F on the has_demographic == 1 subpopulation for all attributes.

    Parameters
    ----------
    panel : pd.DataFrame
        Cutoff panel enriched with demographic features.
    attrs : Sequence[str], optional
        List of attribute names without prefix (e.g. 'age', 'income') or with prefix 'demo_<attr>'.
    thr : FindingThresholds, optional
        Threshold configuration. Defaults to FindingThresholds(min_row_share=0.02).

    Returns
    -------
    pd.DataFrame
        Long-format table with columns: attribute, level, n_rows, n_households, row_share,
        observed, expected, excess_pp, same_sign_share, verdict, verdict_reason.
    """
    if thr is None:
        thr = FindingThresholds(min_row_share=0.02)

    sub = panel[panel["has_demographic"] == 1].copy()
    if sub.empty:
        raise ValueError("Không có dòng nào có has_demographic == 1 trong panel")

    default_attrs = ["age", "income", "homeowner", "marital", "hh_comp", "hh_size", "kids"]
    target_attrs = list(attrs) if attrs is not None else default_attrs

    tables: list[pd.DataFrame] = []

    attr_labels = {
        "age": "Độ tuổi (Age)",
        "income": "Thu nhập (Income)",
        "homeowner": "Nhà ở (Homeowner)",
        "marital": "Hôn nhân (Marital)",
        "hh_comp": "Cơ cấu hộ (HH Comp)",
        "hh_size": "Quy mô hộ (HH Size)",
        "kids": "Số con (Kids)",
    }

    for raw_attr in target_attrs:
        clean_attr = raw_attr.replace("demo_", "")
        col = f"demo_{clean_attr}" if not raw_attr.startswith("demo_") else raw_attr

        if col not in sub.columns:
            raise ValueError(f"Thiếu cột '{col}' trong panel")

        # Remove unused category 'Không có dữ liệu' for clean subpopulation calculation
        if isinstance(sub[col].dtype, pd.CategoricalDtype):
            sub[col] = sub[col].cat.remove_unused_categories()

        t = excess_over_rf(sub, col)
        t["attribute_key"] = clean_attr
        t["attribute"] = attr_labels.get(clean_attr, clean_attr)
        t["level"] = t[col].astype(str)
        tables.append(t)

    combined = pd.concat(tables, ignore_index=True)
    with_verdicts = add_verdicts(combined, thr)

    cols = [
        "attribute",
        "attribute_key",
        "level",
        "n_rows",
        "n_households",
        "row_share",
        "observed",
        "expected",
        "excess_pp",
        "same_sign_share",
        "n_cutoffs_used",
        "verdict",
        "verdict_reason",
    ]
    return with_verdicts[cols]

"""Identifier and candidate-grain profiling."""

from __future__ import annotations

import pandas as pd

from propensity.overview.missing import missing_mask
from propensity.overview.models import (
    GRAIN_SUMMARY_CONTRACT,
    IDENTIFIER_SUMMARY_CONTRACT,
    normalize_result_frame,
)
from propensity.schema.spec import DatasetSpec


def profile_identifiers(
    frame: pd.DataFrame,
    table_name: str,
    spec: DatasetSpec,
) -> pd.DataFrame:
    """Profile cardinality, nulls, uniqueness, and validity of declared identifiers."""
    if not spec.identifier_cols:
        return normalize_result_frame(
            pd.DataFrame(columns=list(IDENTIFIER_SUMMARY_CONTRACT.columns)),
            IDENTIFIER_SUMMARY_CONTRACT,
        )

    n_rows = len(frame)
    rows: list[dict[str, object]] = []

    for col in spec.identifier_cols:
        if col not in frame.columns:
            continue
        m_mask = missing_mask(frame[col])
        n_missing = int(m_mask.sum())
        missing_pct = float(n_missing / n_rows * 100.0) if n_rows > 0 else pd.NA
        non_m = frame.loc[~m_mask, col]
        n_unique_non_missing = int(non_m.nunique(dropna=False))

        if n_rows == 0:
            is_unique = True
            is_valid = False
        else:
            is_unique = bool(n_unique_non_missing == len(non_m) and len(non_m) > 0)
            is_valid = bool(is_unique and n_missing == 0)

        rows.append(
            {
                "table": table_name,
                "column": str(col),
                "n_rows": n_rows,
                "n_missing": n_missing,
                "missing_pct": missing_pct,
                "n_unique_non_missing": n_unique_non_missing,
                "is_unique": is_unique,
                "is_valid": is_valid,
            }
        )

    result = pd.DataFrame(rows)
    return normalize_result_frame(result, IDENTIFIER_SUMMARY_CONTRACT)


def profile_candidate_grain(
    frame: pd.DataFrame,
    table_name: str,
    spec: DatasetSpec,
) -> pd.DataFrame:
    """Validate candidate grain completeness, duplicate groups, excess rows, and validity."""
    if not spec.grain_cols:
        return normalize_result_frame(
            pd.DataFrame(columns=list(GRAIN_SUMMARY_CONTRACT.columns)),
            GRAIN_SUMMARY_CONTRACT,
        )

    n_rows = len(frame)
    grain_cols = list(spec.grain_cols)
    grain_label = ", ".join(grain_cols)

    if n_rows == 0:
        result = pd.DataFrame(
            [
                {
                    "table": table_name,
                    "grain": grain_label,
                    "n_rows": 0,
                    "complete_key_rows": 0,
                    "missing_key_rows": 0,
                    "unique_groups": 0,
                    "duplicate_groups": 0,
                    "excess_rows": 0,
                    "max_rows_per_group": 0,
                    "is_unique": True,
                    "is_valid": False,
                }
            ]
        )
        return normalize_result_frame(result, GRAIN_SUMMARY_CONTRACT)

    # Check for missing values in any grain column
    has_missing = pd.Series(False, index=frame.index)
    for col in grain_cols:
        if col in frame.columns:
            has_missing |= missing_mask(frame[col])
        else:
            has_missing = pd.Series(True, index=frame.index)
            break

    complete_key_rows = int((~has_missing).sum())
    missing_key_rows = int(has_missing.sum())

    if complete_key_rows == 0:
        unique_groups = 0
        duplicate_groups = 0
        excess_rows = 0
        max_rows_per_group = 0
        is_unique = False
        is_valid = False
    else:
        complete_keys = frame.loc[~has_missing, grain_cols]
        group_sizes = complete_keys.groupby(grain_cols, dropna=False).size()
        unique_groups = int((group_sizes == 1).sum())
        duplicate_groups = int((group_sizes > 1).sum())
        excess_rows = int((group_sizes[group_sizes > 1] - 1).sum())
        max_rows_per_group = int(group_sizes.max())
        is_unique = bool(duplicate_groups == 0)
        is_valid = bool(is_unique and missing_key_rows == 0)

    result = pd.DataFrame(
        [
            {
                "table": table_name,
                "grain": grain_label,
                "n_rows": n_rows,
                "complete_key_rows": complete_key_rows,
                "missing_key_rows": missing_key_rows,
                "unique_groups": unique_groups,
                "duplicate_groups": duplicate_groups,
                "excess_rows": excess_rows,
                "max_rows_per_group": max_rows_per_group,
                "is_unique": is_unique,
                "is_valid": is_valid,
            }
        ]
    )
    return normalize_result_frame(result, GRAIN_SUMMARY_CONTRACT)

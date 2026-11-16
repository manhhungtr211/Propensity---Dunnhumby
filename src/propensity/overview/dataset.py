"""Single-dataset profiler consolidating structure, grain, key, time, and quality."""

from __future__ import annotations

import pandas as pd

from propensity.overview.keys import profile_candidate_grain, profile_identifiers
from propensity.overview.models import (
    PROVENANCE_CONTRACT,
    OverviewReport,
    normalize_result_frame,
)
from propensity.overview.quality import profile_quality
from propensity.overview.structure import profile_columns, profile_table_structure
from propensity.overview.temporal import profile_time_coverage
from propensity.schema.spec import DatasetSpec


def profile_dataset(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    table_name: str | None = None,
) -> OverviewReport:
    """Run all overview checks on one dataset and return a packaged OverviewReport.

    This packages structure, candidate grain, identifiers, time coverage,
    and basic quality into a unified report object without requiring separate calls.
    """
    name = table_name or spec.name
    spec.validate(frame)

    table_summary = profile_table_structure(frame, name)
    column_summary = profile_columns(frame, name, spec)
    identifier_summary = profile_identifiers(frame, name, spec)
    grain_summary = profile_candidate_grain(frame, name, spec)
    time_summary = profile_time_coverage(frame, name, spec)
    quality_summary = profile_quality(frame, name, spec)

    provenance_row = pd.DataFrame(
        [
            {
                "table": name,
                "path": ":memory:",
                "size_bytes": int(frame.memory_usage(deep=True).sum())
                if len(frame.columns) > 0
                else 0,
                "scan_mode": "memory",
                "chunk_rows": pd.NA,
                "exact": True,
                "rows_scanned": len(frame),
                "spill_partitions": pd.NA,
                "max_chunk_rows": len(frame),
                "max_partition_rows": pd.NA,
            }
        ]
    )
    provenance = normalize_result_frame(provenance_row, PROVENANCE_CONTRACT)

    return OverviewReport(
        table_summary=table_summary,
        column_summary=column_summary,
        identifier_summary=identifier_summary,
        grain_summary=grain_summary,
        time_summary=time_summary,
        quality_summary=quality_summary,
        provenance=provenance,
    )

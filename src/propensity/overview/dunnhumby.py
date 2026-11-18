"""Orchestration workflow for Dunnhumby source datasets."""

from __future__ import annotations

import gc
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from propensity.io.csv import read_csv
from propensity.overview.dataset import profile_dataset
from propensity.overview.models import (
    CATEGORICAL_SUMMARY_CONTRACT,
    COLUMN_SUMMARY_CONTRACT,
    GRAIN_SUMMARY_CONTRACT,
    IDENTIFIER_SUMMARY_CONTRACT,
    NUMERIC_SUMMARY_CONTRACT,
    PROVENANCE_CONTRACT,
    QUALITY_SUMMARY_CONTRACT,
    TABLE_SUMMARY_CONTRACT,
    TIME_SUMMARY_CONTRACT,
    OverviewReport,
    normalize_result_frame,
)
from propensity.schema.dunnhumby import DUNNHUMBY_CATALOG


def build_dunnhumby_overview(
    raw_dir: Path | str,
    tables: Sequence[str] | None = None,
    chunk_rows: int = 250_000,
    spill_partitions: int = 64,
) -> OverviewReport:
    """Scan and profile Dunnhumby source tables sequentially.

    Returns a single consolidated OverviewReport grouping all profiled tables.
    Processes one table at a time to keep peak memory bounded.
    """
    raw_path = Path(raw_dir)
    target_tables = (
        tuple(tables) if tables is not None else tuple(DUNNHUMBY_CATALOG.keys())
    )

    table_summaries: list[pd.DataFrame] = []
    column_summaries: list[pd.DataFrame] = []
    identifier_summaries: list[pd.DataFrame] = []
    grain_summaries: list[pd.DataFrame] = []
    time_summaries: list[pd.DataFrame] = []
    quality_summaries: list[pd.DataFrame] = []
    numeric_summaries: list[pd.DataFrame] = []
    categorical_summaries: list[pd.DataFrame] = []
    provenances: list[pd.DataFrame] = []

    # Support aliases matching exact file stem names
    aliases = {
        "hh_demographic": "demographic",
        "transaction_data": "transaction",
        "causal_data": "causal",
    }
    canonical_tables = tuple(aliases.get(t, t) for t in target_tables)

    for tbl_name in canonical_tables:
        if tbl_name not in DUNNHUMBY_CATALOG:
            raise KeyError(f"Unknown Dunnhumby table: {tbl_name}")

        info = DUNNHUMBY_CATALOG[tbl_name]
        csv_file = raw_path / info.filename
        if not csv_file.exists():
            raise FileNotFoundError(f"Source file not found: {csv_file}")

        file_size = csv_file.stat().st_size

        # Load table and profile
        frame = read_csv(csv_file, spec=info.spec)
        report = profile_dataset(frame, spec=info.spec, table_name=tbl_name)

        # Record provenance for this table file
        prov_row = pd.DataFrame(
            [
                {
                    "table": tbl_name,
                    "path": str(csv_file),
                    "size_bytes": file_size,
                    "scan_mode": "file",
                    "chunk_rows": chunk_rows,
                    "exact": True,
                    "rows_scanned": len(frame),
                    "spill_partitions": spill_partitions,
                    "max_chunk_rows": len(frame),
                    "max_partition_rows": pd.NA,
                }
            ]
        )
        prov = normalize_result_frame(prov_row, PROVENANCE_CONTRACT)

        table_summaries.append(report.table_summary)
        column_summaries.append(report.column_summary)
        identifier_summaries.append(report.identifier_summary)
        grain_summaries.append(report.grain_summary)
        time_summaries.append(report.time_summary)
        quality_summaries.append(report.quality_summary)
        numeric_summaries.append(report.numeric_summary)
        categorical_summaries.append(report.categorical_summary)
        provenances.append(prov)

        # Release memory before reading next table
        del frame, report
        gc.collect()

    def _concat_or_empty(frames: list[pd.DataFrame], contract: object) -> pd.DataFrame:
        if not frames:
            return normalize_result_frame(pd.DataFrame(), contract)  # type: ignore[arg-type]
        combined = pd.concat(frames, ignore_index=True)
        return normalize_result_frame(combined, contract)  # type: ignore[arg-type]

    return OverviewReport(
        table_summary=_concat_or_empty(table_summaries, TABLE_SUMMARY_CONTRACT),
        column_summary=_concat_or_empty(column_summaries, COLUMN_SUMMARY_CONTRACT),
        identifier_summary=_concat_or_empty(
            identifier_summaries, IDENTIFIER_SUMMARY_CONTRACT
        ),
        grain_summary=_concat_or_empty(grain_summaries, GRAIN_SUMMARY_CONTRACT),
        time_summary=_concat_or_empty(time_summaries, TIME_SUMMARY_CONTRACT),
        quality_summary=_concat_or_empty(quality_summaries, QUALITY_SUMMARY_CONTRACT),
        numeric_summary=_concat_or_empty(numeric_summaries, NUMERIC_SUMMARY_CONTRACT),
        categorical_summary=_concat_or_empty(
            categorical_summaries, CATEGORICAL_SUMMARY_CONTRACT
        ),
        provenance=_concat_or_empty(provenances, PROVENANCE_CONTRACT),
    )

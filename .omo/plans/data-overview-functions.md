# data-overview-functions - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->
<!-- Plain English for a non-engineer: NO file paths, NO todo numbers, NO wave/agent/tool names. -->

**Who this is for and what changes for them:** The analyst/reviewer can understand every Dunnhumby source table through one reusable Python workflow, and future programmers can rerun the same checks after a data refresh without rebuilding notebook logic.

**What you'll get:** Plain pandas result tables for structure, identifiers, candidate grain, time coverage, basic quality, and scan provenance across all eight sources; automated tests; a thin notebook; and a thin command-line runner.

**Why this approach:** Business semantics use one classification system, while ordinal, clock, datetime, and interval behavior is declared explicitly. Large files are scanned sequentially and spill exact global comparisons to temporary partitions so results remain exact without retaining every source table in memory.

**What it will NOT do:** It will not engineer features, build a target/model, apply deep business or relationship rules, draw charts, silently deduplicate rows, or invent calendar dates.

**Effort:** Large
<!-- Effort is exactly ONE band, never hours/days. Quick = single edit, minutes of agent work; Short = one focused change, a few files; Medium = multi-file feature in one session; Large = several waves, one long session; XL = multi-session or architectural work. A written duration is rewritten to a band. -->
**Risk:** Medium - exact duplicate/grain analysis over the 36.8-million-row causal table needs careful spill-partition correctness and full-data QA.
**Decisions to sanity-check:** Remove the secondary analysis enum; include all eight tables; make exact metrics the default; keep deep business and referential checks for a later phase.

Your next move: execute this plan in a separate worker session, or request a high-accuracy plan review first. Full execution detail follows below.

---

> TL;DR (machine): Large effort, medium risk; 11 implementation todos deliver a pandas overview package, exact chunk/spill orchestration, eight-table contracts, tests, CLI, and a zero-logic notebook.

## Scope
### Affected user and ideal state
**Affected user:** The analyst/reviewer who must understand the Dunnhumby source before feature engineering, plus future programmers who rerun the same checks after a data refresh. Today the active overview notebook is empty and the reusable EDA implementation is absent; after execution they call one packaged pandas workflow and inspect its raw result tables.

| Row | Statement | Reason |
| --- | --- | --- |
| IS-1 | One packaged Python call profiles all eight raw Dunnhumby tables and returns plain pandas DataFrames. | Analysis logic must not be maintained in notebooks. |
| IS-2 | Structure, candidate grain, identifiers, time coverage, and basic quality use stable, documented output columns across tables. | Results must be comparable and reusable. |
| IS-3 | Grain violations and exact duplicates are reported without silently deduplicating source rows. | Coupon and causal data already violate candidate uniqueness. |
| IS-4 | DAY/WEEK_NO stay ordinal, TRANS_TIME stays clock-coded, and START_DAY/END_DAY remain an interval. | Artificial calendar conversion would create false business meaning. |
| IS-5 | The 696 MB causal and 142 MB transaction sources run sequentially with bounded peak memory and explicit exact/approximate provenance. | The workflow must run on the actual repository data. |
| IS-6 | The active overview notebook contains imports, path/config setup, one package call, and direct result display only. | No notebook-only functions or presentation code. |
| IS-7 | Automated tests cover happy and failure paths for every public profiler and all eight source contracts. | The current pandas baseline has no tests directory. |
| GAP-1 | No working overview package exists; descriptive files import deleted APIs. | Blocks IS-1 and IS-2. |
| GAP-2 | Only transaction/product specs exist and the current time model cannot express ordinal, clock, and interval fields together. | Blocks IS-2 and IS-4. |
| GAP-3 | The active notebook is empty while historical drafts contain scattered Polars-era calculations. | Blocks IS-1 and IS-6. |
| GAP-4 | No tests directory exists despite pytest configuration. | Blocks IS-7. |
| GAP-5 | Configured NA tokens are not passed through the CSV boundary. | Makes missingness inconsistent and blocks IS-2. |
| GAP-6 | No exact cross-chunk duplicate/grain reducer or scan provenance exists for large CSVs. | Blocks IS-3 and IS-5. |

### Must have
- Generic pandas functions for structure, identifiers, candidate grain, time coverage, and basic quality.
- Simple Dunnhumby configuration for all eight CSVs: filename, required columns, identifiers, candidate grain, row meaning, string codes, and explicit time semantics.
- `ColumnRole` as the only column classification enum; remove `AnalysisKind` and every active-source import/export/override that depends on it.
- Exact metric definitions: distinct excludes missing; key validity means zero missing-key rows plus uniqueness; duplicate groups and excess rows are separate; all-missing is not constant; empty-input percentages are `pd.NA`.
- Pandas chunk scans for additive metrics and spill partitions for exact global grain/full-row duplicate comparisons across chunk boundaries. Hashes may route rows to partitions but never prove equality.
- A compact report object whose attributes are plain `pd.DataFrame` instances with fixed columns, plus per-table provenance.
- Tests-after with deterministic DataFrame fixtures, cross-chunk duplicate fixtures, small real-file integration, installed-package import smoke, notebook AST guard, and an opt-in full-eight-table QA run.

### Must NOT have (guardrails, anti-slop, scope boundaries)
- No feature engineering, target construction, modeling, charts, styling, slide work, deep domain rules, or referential-integrity audit.
- No silent deduplication, mutation of source values, identifier-to-float coercion, approximate metric presented as exact, or artificial calendar dates.
- No local function/class definitions or analytical transformations in `notebooks/00_data_overview.ipynb`.
- No restoration of unrelated legacy entity/target/report APIs and no import through the broken `propensity.eda.descriptive` facade.
- No full scan of the 696 MB causal source in routine unit tests.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: tests-after with pytest; each implementation todo includes its own tests and Given/When/Then cases.
- Static gates: `uv run ruff check src tests scripts`, `uv run ruff format --check src tests scripts`, and `uv run python -m compileall -q src scripts`.
- Routine behavioral gate: `uv run pytest -q`; the full raw-data scan is marked `full_data` and executed explicitly during final QA, not on every unit run.
- Evidence: one task-numbered and one final-check-numbered text/JSON artifact under `.omo/evidence/data-overview-functions/`.

## Execution strategy
### Parallel execution waves
- Wave 1 — contracts and boundaries in parallel: Todos 1-3.
- Wave 2 — five profiler/config responsibilities in parallel after Wave 1: Todos 4-8.
- Wave 3 — three consumers/integration surfaces in parallel against the fixed contracts: Todos 9-11.
- Final verification — F1-F4 in parallel only after all implementation todos pass.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | — | 5, 6, 7, 8, 9, 10 | 2, 3 |
| 2 | — | 4, 7, 8, 9 | 1, 3 |
| 3 | — | 4, 5, 6, 7, 9 | 1, 2 |
| 4 | 2, 3 | 9, 11 | 5, 6, 7, 8 |
| 5 | 1, 3 | 9 | 4, 6, 7, 8 |
| 6 | 1, 3 | 9 | 4, 5, 7, 8 |
| 7 | 1, 2, 3 | 9 | 4, 5, 6, 8 |
| 8 | 1, 2 | 9 | 4, 5, 6, 7 |
| 9 | 1-8 | F1-F4 | 10, 11 |
| 10 | 1, 2 | F1-F4 | 9, 11 |
| 11 | 1, 4 | F1-F4 | 9, 10 |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->
- [ ] 1. Define stable overview result contracts
  - What to do: create `src/propensity/overview/models.py` with frozen/slotted report value objects grouping `table_summary`, `column_summary`, `identifier_summary`, `grain_summary`, `time_summary`, `quality_summary`, and `provenance`, each a plain pandas DataFrame. Define the exact ordered column tuple for every result table and empty constructors that preserve those columns. Add `tests/test_overview_models.py`.
  - Must NOT do: do not add presentation methods, charts, generic raw dict returns, or duplicate metric computation in the models.
  - Metric contract: empty result frames retain fixed columns; percentages use nullable `Float64`; counts use nullable `Int64`; provenance includes table, path, size bytes, scan mode, chunk rows, exact flag, and rows scanned, without a nondeterministic timestamp.
  - Closes: GAP-1, GAP-4.
  - Parallelization: Wave 1 | Blocked by: none | Blocks: 5, 6, 7, 8, 9, 10.
  - References: `src/propensity/schema/spec.py:41-54`; `.omo/drafts/data-overview-functions.md` IS-1/IS-2; Metis metric-definition finding.
  - Acceptance criteria: `uv run pytest tests/test_overview_models.py -q` passes and asserts exact column order/dtypes for populated and empty reports; `uv run ruff check src/propensity/overview/models.py tests/test_overview_models.py` passes.
  - QA scenarios: happy — construct all seven result frames and access them through the report; failure — constructing a report with a missing required result column raises a typed contract error. Evidence: `.omo/evidence/data-overview-functions/task-1.txt`.
  - Commit: Y | `feat(overview): define stable report contracts`
  - Recommended task executor category: `quick` — isolated value objects and tests.

- [ ] 2. Collapse schema semantics to ColumnRole plus explicit time metadata
  - What to do: update `src/propensity/schema/spec.py` with simple tuple fields `string_cols`, `ordinal_time_cols`, `clock_time_cols`, and `interval_cols` (pairs); retain existing identifiers/datetimes and validate blank, duplicated, overlapping, missing interval endpoints, and identical interval endpoints. Update `src/propensity/schema/infer.py` precedence to entity → target → identifier → datetime → any explicit non-calendar time → physical dtype. Remove `AnalysisKind`, `src/propensity/schema/features.py` if empty, its export from `schema/__init__.py`, `TRANSACTION_ANALYSIS_OVERRIDES`, and all active-source imports/usages. Add `tests/test_schema_overview.py`.
  - Must NOT do: do not infer time units from column names; do not modify historical notebooks; do not introduce a replacement enum.
  - Closes: GAP-2, GAP-4.
  - Parallelization: Wave 1 | Blocked by: none | Blocks: 4, 7, 8, 9.
  - References: `src/propensity/schema/spec.py:41-66`; `src/propensity/schema/infer.py`; `src/propensity/schema/features.py`; `src/propensity/schema/__init__.py`; `src/propensity/schema/dunnhumby.py:40-46`; approved user decision to remove `AnalysisKind`; Metis time-precedence/import findings.
  - Acceptance criteria: `uv run pytest tests/test_schema_overview.py -q` proves valid and invalid overlaps, interval pairs, and role precedence; `rg -n "AnalysisKind|TRANSACTION_ANALYSIS_OVERRIDES" src` returns no matches; `uv run python -c "from propensity.schema import ColumnRole, DatasetSpec"` succeeds.
  - QA scenarios: happy — DAY/WEEK_NO/TRANS_TIME resolve to `ColumnRole.TIME` while a genuine parsed date resolves to `DATETIME`; failure — the same column declared in datetime and ordinal lists raises `DatasetSpecError`. Evidence: `.omo/evidence/data-overview-functions/task-2.txt`.
  - Commit: Y | `refactor(schema): simplify overview time metadata`
  - Recommended task executor category: `unspecified-high` — cross-module public schema migration.

- [ ] 3. Activate consistent missing-value and CSV dtype handling
  - What to do: add `src/propensity/overview/missing.py` with reusable missing masks that combine pandas nulls with whitespace-only strings without mutating source values. Update `src/propensity/io/csv.py` to pass configured NA tokens as additions to pandas defaults, apply nullable `Int64` only to identifier columns, `StringDtype` to declared string-code columns, and preserve datetime parsing. Add `tests/test_missing.py` and `tests/test_csv_io.py` using temporary CSVs.
  - Must NOT do: do not treat code `"0"` as missing, strip/replace nonmissing values, or force every numeric-looking string to an integer.
  - Closes: GAP-4, GAP-5.
  - Parallelization: Wave 1 | Blocked by: none | Blocks: 4, 5, 6, 7, 9.
  - References: `src/propensity/config.py:9-17`; `src/propensity/io/csv.py:12-30`; `src/propensity/schema/spec.py:51-54`; Metis parsing/missingness finding.
  - Acceptance criteria: `uv run pytest tests/test_missing.py tests/test_csv_io.py -q` passes; tests distinguish null, configured token, whitespace-only, empty string, and literal `"0"`, and assert input frames are unchanged.
  - QA scenarios: happy — identifier stays nullable Int64 and causal display/mailer stay pandas strings across differently shaped CSV chunks; failure — missing required columns still raise `DatasetSpecError` at the boundary. Evidence: `.omo/evidence/data-overview-functions/task-3.txt`.
  - Commit: Y | `fix(io): apply configured missing and string semantics`
  - Recommended task executor category: `quick` — focused boundary/helper change.

- [ ] 4. Configure all eight Dunnhumby source tables
  - What to do: replace the partial constants in `src/propensity/schema/dunnhumby.py` with immutable specs for transaction, product, demographic, campaign description, campaign assignment, coupon, coupon redemption, and causal; add an immutable table-name → filename/spec/row-meaning catalog. Encode candidate grains exactly: transaction `(BASKET_ID, PRODUCT_ID)`, product `(PRODUCT_ID)`, demographic `(household_key)`, campaign assignment `(CAMPAIGN, household_key)`, campaign description `(CAMPAIGN)`, coupon `(COUPON_UPC, PRODUCT_ID, CAMPAIGN)`, redemption `(household_key, DAY, COUPON_UPC, CAMPAIGN)`, causal `(PRODUCT_ID, STORE_ID, WEEK_NO)`. Encode DAY/WEEK_NO as ordinal with explicit units, TRANS_TIME as clock, START_DAY/END_DAY as an ordinal interval, and display/mailer as strings. Add `tests/test_dunnhumby_specs.py` that checks actual CSV headers only, not full scans.
  - Must NOT do: do not claim candidate grains are business-key truth or unique; do not add relationship/FK rules.
  - Closes: GAP-2, GAP-4.
  - Parallelization: Wave 2 | Blocked by: 2, 3 | Blocks: 9, 11.
  - References: all headers under `data/raw/Dunnhumby/*.csv`; `src/propensity/schema/dunnhumby.py:9-38`; `data/processed/data_quality/duplicate_summary.csv`; draft findings for verified small-table grains.
  - Acceptance criteria: `uv run pytest tests/test_dunnhumby_specs.py -q` passes and asserts eight catalog entries, filename existence, required-column inclusion, explicit time units, and candidate grains against real headers.
  - QA scenarios: happy — every raw header satisfies its spec without reading data rows; failure — a temporary copy with one configured identifier header removed fails clearly with table and column names. Evidence: `.omo/evidence/data-overview-functions/task-4.txt`.
  - Commit: Y | `feat(schema): configure all dunnhumby tables`
  - Recommended task executor category: `unspecified-high` — eight related source contracts with cross-table consistency.

- [ ] 5. Implement reusable structure profiling
  - What to do: create `src/propensity/overview/structure.py` with `profile_table_structure` and `profile_columns`. Return stable table/column result frames using `resolve_schema`, including rows, columns, memory bytes for in-memory frames, dtype, semantic role, required flag, and configured time/identifier/grain membership.
  - Must NOT do: do not compute quality, uniqueness, or time ranges in this module; do not format output.
  - Closes: GAP-1, GAP-4.
  - Parallelization: Wave 2 | Blocked by: 1, 3 | Blocks: 9.
  - References: `src/propensity/schema/infer.py`; `src/propensity/schema/spec.py:68-95`; result contracts from Todo 1.
  - Acceptance criteria: `uv run pytest tests/test_overview_structure.py -q` passes for mixed pandas dtypes, optional entity/time metadata, zero-row frames, and required-column flags.
  - QA scenarios: happy — mixed transaction-like frame produces one table row and one row per column in source order; failure — spec validation reports a missing required column before profiling. Evidence: `.omo/evidence/data-overview-functions/task-5.txt`.
  - Commit: Y | `feat(overview): add structure profiling`
  - Recommended task executor category: `quick` — one pure profiling responsibility.

- [ ] 6. Implement basic quality profiling
  - What to do: create `src/propensity/overview/quality.py` with column missing/blank metrics and table exact-duplicate metrics for in-memory frames. Define `n_missing`, `missing_pct`, `n_unique_non_missing`, `is_all_missing`, and `is_constant`; all-missing must report `is_constant=False`, and empty-frame percentages must be `pd.NA`.
  - Must NOT do: do not include transaction business rules, FK coverage, outlier thresholds, or source mutation.
  - Closes: GAP-1, GAP-4, GAP-5.
  - Parallelization: Wave 2 | Blocked by: 1, 3 | Blocks: 9.
  - References: `src/propensity/config.py:9-17`; `data/processed/data_quality/table_quality_overview.csv`; `null_summary.csv`; Metis metric-definition finding; Todo 3 missing mask.
  - Acceptance criteria: `uv run pytest tests/test_overview_quality.py -q` passes for nulls, whitespace, `"0"`, duplicate rows, all-missing, constant, mixed, and empty frames.
  - QA scenarios: happy — duplicate and missing counts match independently constructed expected values; failure — an all-missing column is never mislabeled constant and empty percentage is not zero. Evidence: `.omo/evidence/data-overview-functions/task-6.txt`.
  - Commit: Y | `feat(overview): add basic quality profiling`
  - Recommended task executor category: `quick` — pure metrics with explicit edge contracts.

- [ ] 7. Implement identifier and candidate-grain profiling
  - What to do: create `src/propensity/overview/keys.py` with identifier summaries and candidate-grain validation. Distinct counts exclude missing; grain output includes grain label, total rows, complete-key rows, missing-key rows, unique groups, duplicate groups, excess rows, max rows/group, `is_unique`, and `is_valid = is_unique and missing_key_rows == 0`.
  - Must NOT do: do not raise merely because a candidate grain is nonunique; do not deduplicate or use hash equality as proof.
  - Closes: GAP-1, GAP-3, GAP-4.
  - Parallelization: Wave 2 | Blocked by: 1, 2, 3 | Blocks: 9.
  - References: `data/processed/data_quality/duplicate_summary.csv`; approved candidate-grain semantics; Metis metric-definition/global-exactness findings; Todo 1 contracts.
  - Acceptance criteria: `uv run pytest tests/test_overview_keys.py -q` passes for unique, duplicate, missing-key, all-missing, and empty frames and independently asserts duplicate groups versus excess rows.
  - QA scenarios: happy — coupon-like repeated keys yield a nonunique verdict and correct group/excess counts without exception; failure — a unique set with one missing key is `is_unique=True` but `is_valid=False`. Evidence: `.omo/evidence/data-overview-functions/task-7.txt`.
  - Commit: Y | `feat(overview): add key and grain profiling`
  - Recommended task executor category: `unspecified-high` — correctness-sensitive composite-key semantics.

- [ ] 8. Implement semantic time coverage profiling
  - What to do: create `src/propensity/overview/temporal.py` producing stable rows for datetime, ordinal, clock, and interval configurations. Point coverage includes semantic kind, declared unit, min, max, observed distinct periods, span, and missing count. Interval coverage includes start/end labels, min start, max end, missing endpoint rows, and start-after-end rows. Validate clock codes as HHMM without converting them to durations.
  - Must NOT do: do not map Dunnhumby DAY to an epoch, infer units from names, aggregate TRANS_TIME as continuous duration, or require tables to have time fields.
  - Closes: GAP-1, GAP-2, GAP-4.
  - Parallelization: Wave 2 | Blocked by: 1, 2 | Blocks: 9.
  - References: transaction header; campaign_desc header; `src/propensity/schema/dunnhumby.py:19-20,40-45`; cached evidence DAY 1-711/WEEK_NO 1-102/TRANS_TIME 0-2359; Metis time-shape finding.
  - Acceptance criteria: `uv run pytest tests/test_overview_temporal.py -q` passes for all four semantic kinds, no-time tables, missing values, invalid clock code, reversed interval, and empty frames.
  - QA scenarios: happy — DAY and WEEK_NO remain ordinal while START_DAY/END_DAY produce one interval row; failure — `2360` is counted invalid as HHMM and no calendar date column appears. Evidence: `.omo/evidence/data-overview-functions/task-8.txt`.
  - Commit: Y | `feat(overview): add semantic time coverage`
  - Recommended task executor category: `unspecified-high` — multiple semantic variants with strict no-fake-date invariant.

- [ ] 9. Build exact, memory-bounded CSV orchestration
  - What to do: create `src/propensity/overview/scan.py` and `src/propensity/overview/dunnhumby.py`. `build_dunnhumby_overview(raw_dir, *, chunk_rows=250_000, spill_partitions=64)` must process one table at a time. Aggregate additive metrics per pandas chunk. For global full-row duplicates and candidate-grain counts, route original values into temporary pyarrow parquet partition files using stable pandas hashes only as partition selectors, then compare/group exact original values within each partition; merge compact results. Use context-managed temp storage and emit provenance with exact flag, rows scanned, chunk size, partition count, and maximum chunk/partition rows. Add `tests/test_overview_scan.py` including duplicates split across chunk boundaries and hash-collision-style routing fixtures.
  - Must NOT do: do not sum per-chunk distinct/duplicate counts, trust hash equality, retain all eight frames simultaneously, leak temp files, or fall back silently to approximation.
  - Closes: GAP-1, GAP-6.
  - Parallelization: Wave 3 | Blocked by: 1-8 | Blocks: final verification.
  - References: `src/propensity/io/csv.py`; `pyproject.toml:9`; raw file sizes under `data/raw/Dunnhumby`; Metis global-exactness/topology findings; profiler contracts from Todos 5-8.
  - Acceptance criteria: `uv run pytest tests/test_overview_scan.py -q` passes and proves cross-chunk exactness against a full in-memory pandas oracle; temp directory is empty after success and raised parsing failure; peak provenance never exceeds configured chunk size except a reported spill partition, whose size is recorded.
  - QA scenarios: happy — a multi-file fixture catalog produces exact consolidated result frames matching in-memory profilers; failure — a duplicate key whose two rows occur in different chunks is still counted once as a duplicate group with one excess row, and malformed required columns fail without leftover spill files. Evidence: `.omo/evidence/data-overview-functions/task-9.txt`.
  - Commit: Y | `feat(overview): add exact chunked dunnhumby scan`
  - Recommended task executor category: `deep-low` — exactness and bounded-memory reasoning across chunk/spill boundaries.

- [ ] 10. Publish the focused package API and execution smoke surface
  - What to do: add `src/propensity/overview/__init__.py` exporting only the report types, five profiler families, and Dunnhumby builder; update `src/propensity/__init__.py` with the intended top-level builder/report exports without importing `propensity.eda.descriptive`. Add `scripts/build_data_overview.py` as a thin CLI that accepts raw/output directories, calls the builder, and writes each raw result DataFrame to CSV. Add package/import and CLI fixture tests.
  - Must NOT do: do not re-export legacy entity/target/report names, duplicate profiler logic in the script, or add styling.
  - Closes: GAP-1, GAP-4.
  - Parallelization: Wave 3 | Blocked by: 1, 2 | Blocks: final verification.
  - References: `src/propensity/__init__.py`; `src/propensity/io/__init__.py`; broken facade evidence `src/propensity/eda/descriptive/__init__.py`; existing script packaging conventions in `scripts/build_purchase_feature_dataset.py`; Metis import-integration finding.
  - Acceptance criteria: `uv run pytest tests/test_package.py tests/test_overview_cli.py -q` passes; `uv build` succeeds; an installed wheel imports `propensity` and `propensity.overview` outside the repository; CLI fixture run writes all seven stable CSV outputs.
  - QA scenarios: happy — invoke the script on a temporary eight-table fixture and inspect emitted headers; failure — missing raw directory exits nonzero with the missing path and no partial output directory. Evidence: `.omo/evidence/data-overview-functions/task-10.txt`.
  - Commit: Y | `feat(package): expose data overview workflow`
  - Recommended task executor category: `quick` — thin exports/entrypoint against predetermined interfaces.

- [ ] 11. Make the active notebook a zero-logic consumer
  - What to do: populate `notebooks/00_data_overview.ipynb` with a short title/goal cell, imports/path setup, one `build_dunnhumby_overview` call, and direct display of report attributes. Add `tests/test_notebook_00.py` that parses notebook JSON and Python AST to permit imports, constants/path setup, one package call, assignments, and direct expressions only; reject function/class/lambda definitions and DataFrame transformations.
  - Must NOT do: do not copy analysis code from draft notebooks, add charts/styling, mutate pandas display globals, or edit historical draft notebooks.
  - Closes: GAP-3, GAP-4.
  - Parallelization: Wave 3 | Blocked by: 1, 4 | Blocks: final verification.
  - References: empty `notebooks/00_data_overview.ipynb`; `notebooks/draft_00_data_inventory.ipynb` and `draft_02_data_quality.ipynb` as evidence only; Metis notebook-topology finding.
  - Acceptance criteria: `uv run pytest tests/test_notebook_00.py -q` passes; AST assertions find zero local function/class/lambda nodes and exactly one call to the packaged builder; notebook JSON remains valid nbformat 4.
  - QA scenarios: happy — execute code cells in order from a clean namespace during the full-data QA and obtain each raw result DataFrame; failure — inject a temporary function definition or `.groupby()` call into an in-memory notebook fixture and confirm the guard test rejects it. Evidence: `.omo/evidence/data-overview-functions/task-11.txt`.
  - Commit: Y | `docs(notebook): call packaged overview functions only`
  - Recommended task executor category: `quick` — minimal notebook consumer plus structural guard.

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan compliance audit
  - Verify every Must have/Must NOT have and each Todo 1-11 acceptance command against the live tree; confirm no implementation logic entered the notebook and no `AnalysisKind` remains in active source. Evidence: `.omo/evidence/data-overview-functions/final-1.txt`.
- [ ] F2. Code quality review
  - Run `uv run ruff check src tests scripts`, `uv run ruff format --check src tests scripts`, `uv run python -m compileall -q src scripts`, `uv run pytest -q`, and inspect changed files for single responsibility, typed public APIs, no silent approximation, and no module above 250 pure LOC without an explicit justified split. Evidence: `.omo/evidence/data-overview-functions/final-2.txt`.
- [ ] F3. Real manual QA
  - From a clean process, execute `scripts/build_data_overview.py` against all eight real files into a temporary output directory; separately execute notebook code cells in order. Record command, runtime, output row counts, exact flags, maximum chunk/partition rows, and peak process memory. Verify observed anchors: 8 tables; transaction 2,595,732 rows; product 92,353 rows; coupon 124,548 rows and 5,164 exact excess duplicates; causal 36,786,524 rows; transaction DAY 1-711; campaign interval 224-719. Any mismatch is investigated, not normalized to cached values. Evidence: `.omo/evidence/data-overview-functions/final-3.json`.
- [ ] F4. Ideal-state fidelity
  - Map IS-1 through IS-7 one-to-one to delivered outputs and evidence, including exact cross-chunk duplicates, time semantics, bounded-memory provenance, thin notebook AST, and automated tests. Any shortfall becomes a new numbered implementation todo rather than a note. Evidence: `.omo/evidence/data-overview-functions/final-4.txt`.

## Commit strategy
- One atomic commit per implementation todo using the commit text specified in that todo; do not commit `.omo/evidence/`, generated overview CSVs, raw data, notebook execution outputs, or unrelated existing changes.
- Before each commit run that todo's targeted tests and ruff on its owned paths; before handoff run the full final gates.
- Preserve historical draft notebooks and raw/processed data exactly.

## Success criteria
> One row per IS row. The plan is complete only when every IS row has a delivering todo and a proving QA scenario; F4 checks the delivered behavior against these rows 1:1, and a shortfall becomes new `- [ ] N.` rows, never a note.
| IS | Delivering todo(s) | Proving QA scenario | Evidence |
| --- | --- | --- | --- |
| IS-1 | 1, 5-10 | Full eight-table builder and CLI output | `.omo/evidence/data-overview-functions/final-3.json` |
| IS-2 | 1, 4-8 | Stable schemas across all result frames | `.omo/evidence/data-overview-functions/final-4.txt` |
| IS-3 | 7, 9 | Cross-chunk duplicate fixture plus coupon/causal full scan | `.omo/evidence/data-overview-functions/task-9.txt`, `final-3.json` |
| IS-4 | 2, 4, 8 | Ordinal/clock/interval semantic QA with no fake dates | `.omo/evidence/data-overview-functions/task-8.txt`, `final-4.txt` |
| IS-5 | 9 | Full real-data scan with chunk/partition and peak-memory evidence | `.omo/evidence/data-overview-functions/final-3.json` |
| IS-6 | 11 | Notebook AST guard and clean-namespace execution | `.omo/evidence/data-overview-functions/task-11.txt`, `final-3.json` |
| IS-7 | 1-11 | Targeted tests plus full pytest/static gates | `.omo/evidence/data-overview-functions/final-2.txt` |

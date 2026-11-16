---
slug: data-overview-functions
status: plan-written
intent: clear
review_required: false
pending-action: execute .omo/plans/data-overview-functions.md or run optional high-accuracy review
approach: Build a small pandas overview package with reusable table-level functions, one Dunnhumby orchestration function, compact raw DataFrame outputs, tests-after coverage, and a notebook that only calls the package.
---

# Draft: data-overview-functions

## Affected user and ideal state
- Affected user: the analyst/reviewer who needs to understand Dunnhumby before feature engineering, and future programmers who rerun the same checks after data updates.
- IS-1 | One packaged Python call profiles all eight raw Dunnhumby tables and returns plain pandas DataFrames | The analyst should not maintain analysis logic in notebooks.
- IS-2 | Structure, candidate grain, identifiers, time coverage, and basic quality are reported with one consistent schema | Results must be comparable across tables and reusable by later reporting.
- IS-3 | Grain violations and exact duplicates are surfaced, never silently deduplicated | Coupon and causal data already demonstrate that a candidate grain can be violated.
- IS-4 | DAY/WEEK_NO remain ordinal indices, TRANS_TIME remains clock coding, and campaign START_DAY/END_DAY remain an interval | Fake calendar dates would create false business meaning.
- IS-5 | The 696 MB causal file and 142 MB transaction file are analyzed sequentially with bounded peak memory and explicit exact/approximation provenance | The overview must be runnable on the actual repository data.
- IS-6 | `notebooks/00_data_overview.ipynb` contains imports, one packaged function call, and raw result display only | No scattered notebook-only functions or presentation styling.
- IS-7 | Automated tests prove happy and failure paths for each reusable function and the Dunnhumby configuration | The current checkout has no test suite for the new pandas baseline.
- GAP-1 | `src/propensity/eda` has no working overview implementation; only compatibility shims import missing modules | closes IS-1, IS-2.
- GAP-2 | `DatasetSpec` covers transaction/product only and has a single primary time field | closes IS-2, IS-4.
- GAP-3 | Existing draft notebooks contain Polars-era local calculations while the active overview notebook is empty | closes IS-1, IS-6.
- GAP-4 | No current `tests/` directory exists even though pytest is configured to use it | closes IS-7.
- GAP-5 | `DatasetSpec.na_tokens` is not passed through `read_csv`, so configured missing-value semantics are inactive | closes IS-2.
- GAP-6 | No packaged strategy records scope/provenance or controls the cost of exact metrics on large files | closes IS-3, IS-5.

## Components (topology ledger)
- C1 | Eight-table Dunnhumby configuration defines file, identifiers, candidate grain, row meaning, and time semantics | active | `data/raw/Dunnhumby/*.csv`; `src/propensity/schema/dunnhumby.py:9-46`
- C2 | Structure and column profiling functions return raw pandas summary tables | active | `src/propensity/schema/infer.py`; missing implementation evidenced by `src/propensity/eda/descriptive/dataset.py:3-13`
- C3 | Identifier and candidate-grain functions report cardinality, missing keys, duplicate groups, excess rows, and uniqueness verdict | active | `data/processed/data_quality/duplicate_summary.csv`
- C4 | Time coverage functions handle ordinal points, clock codes, and start/end intervals without calendar conversion | active | transaction/campaign/causal headers; `src/propensity/schema/dunnhumby.py:19-20,40-45`
- C5 | Basic quality functions report null/blank, exact duplicate, constant, and all-missing metrics | active | `data/processed/data_quality/table_quality_overview.csv`; `null_summary.csv`
- C6 | Sequential orchestrator, public exports, tests, and thin notebook make the functions directly usable | active | `notebooks/00_data_overview.ipynb` is empty; `pyproject.toml:20-21`

## Open assumptions (announced defaults)
- Output surface | Each profiler returns a plain `pd.DataFrame`; one top-level report object only groups the named result frames, without styling | matches the request to use the original function form | yes
- Dataset scope | Include all eight Dunnhumby CSV tables, not transaction only | “tổng quan dữ liệu” and the raw directory define the complete source set | yes
- Quality depth | Include structural/basic checks only: missing/blank, exact duplicate, constants, key/grain validity; exclude business-domain and referential rules | keeps this phase before deep quality analysis | yes
- Accuracy | Exact row/missing/duplicate/grain results by default; any approximation must be opt-in and labeled | grain claims must not rely on approximate cardinality | yes
- Scale | Process one table at a time, use pandas chunking for aggregatable metrics, and isolate expensive exact grain passes to configured key columns | causal is 695,858,427 bytes | yes
- Missing values | Activate `DatasetSpec.na_tokens` and trim whitespace only for missingness evaluation; do not mutate source values | current config exists but is unused | yes
- Test strategy | Tests-after with small DataFrame fixtures plus one lightweight real-file integration on small Dunnhumby tables; no full 696 MB scan in routine unit tests | fast deterministic suite with direct production evidence | yes
- Notebook | Replace the empty active notebook only; keep historical draft notebooks as evidence and out of implementation scope | avoids rewriting old narrative artifacts | yes

## Findings (cited - path:lines)
- pandas and pyarrow are the runtime stack: `pyproject.toml:5-14`.
- The reusable input boundary already applies nullable `Int64` identifiers but ignores configured NA tokens: `src/propensity/io/csv.py:12-30`.
- `DatasetSpec` already carries grain, identifier, datetime, and NA metadata: `src/propensity/schema/spec.py:41-54`.
- Only transaction and product have current concrete specifications: `src/propensity/schema/dunnhumby.py:16-38`.
- The descriptive files are broken compatibility imports, not implementations: `src/propensity/eda/descriptive/dataset.py:3-13`, `quality.py:3-4`, `temporal.py:3-8`.
- Active `notebooks/00_data_overview.ipynb` has zero cells; the Polars-era analysis lives in draft notebooks.
- Actual source sizes are 141,742,346 bytes for transactions and 695,858,427 bytes for causal; all eight headers were verified directly under `data/raw/Dunnhumby/`.
- Direct small-table verification: product 92,353 unique PRODUCT_IDs; demographic 801 unique households; campaign_desc 30 unique campaigns with START_DAY 224-659 and END_DAY 264-719; coupon has 5,164 excess duplicate rows at `(COUPON_UPC, PRODUCT_ID, CAMPAIGN)`; redemption DAY spans 225-704.
- Cached, not freshly rescanned large-table evidence: transaction has 2,595,732 rows and DAY 1-711; causal has 36,786,524 rows and 93 observed weeks. The implementation must recompute rather than trust notebook cache.

## Decisions (with rationale)
- Create a focused overview package instead of restoring the deleted broad EDA framework; this phase needs five explicit responsibilities, not entity/target/model analysis.
- Keep `ColumnRole` for business semantics and remove `AnalysisKind`; represent special time handling directly with explicit ordinal, clock, datetime, and interval column configuration so there is only one classification system.
- Preserve semantic time types in configuration: ordinal point, clock code, and interval pair. Do not derive artificial calendar dates.
- Treat every declared grain as a candidate plus observed verdict; uniqueness failure is valid output, not an exception.
- Return compact report tables and provenance rather than formatted displays or charts.
- Keep notebook logic to orchestration and direct display of function results.

## Scope IN
- Generic pandas functions for structure, keys, candidate grain, time coverage, and basic quality.
- Dunnhumby configuration for all eight raw CSV files.
- Memory-aware sequential orchestration and provenance fields.
- Public package exports, tests, and a thin `00_data_overview.ipynb` consumer.

## Scope OUT (Must NOT have)
- No feature engineering, target construction, model training, or feature-target analysis.
- No charts, color formatting, slide design, or narrative storytelling.
- No deep business rules or cross-table referential-integrity audit in this phase.
- No silent deduplication, type coercion that changes identifiers, or fake calendar conversion.
- No local analysis functions inside notebooks.
- No restoration of unrelated legacy entity/target/report APIs.

## Open questions
- None blocking. The defaults above are reversible and will be accepted or changed at the approval gate.

## Approval gate
status: approved-plan-written
approach: Plan a focused pandas overview package with five reusable profiler families, an eight-table Dunnhumby configuration, sequential large-file orchestration, tests-after QA, and a notebook limited to calling/displaying packaged results.
next action after approval: create `.omo/plans/data-overview-functions.md`, run mandatory Metis gap analysis, then append decision-complete todos; do not implement.

approval received: user replied "oke" and later "continue" after the Metis retry notice.
metis result: gaps found and folded into the plan — explicit time precedence, exact metric definitions, disk-spill global exactness, parsing/missingness policy, active-source import cleanup, full-data and notebook QA.
plan path: `.omo/plans/data-overview-functions.md`

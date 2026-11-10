---
slug: eda-module-responsibilities
status: awaiting-approval
intent: clear
review_required: false
pending-action: write .omo/plans/eda-module-responsibilities.md
approach: Keep schema as the lightweight WHAT contract; implement table-level understanding one table at a time, beginning with transaction_data; add grain_cols only; treat DAY as an ordinal day index; stabilize Polars loading; preserve compatibility imports; defer relationships, target, report, and speculative profiling expansion.
---

# Draft: eda-module-responsibilities

## Affected user and ideal state

- Affected user: the repository owner/developer, who must know where a new EDA function belongs without rereading the whole package.
- Affected program: notebooks and visualization code importing the current EDA APIs.
- Affected future worker/agent: an implementer that must add one capability without inventing architecture or duplicating logic.

- IS-1 | Every production module answers one named analytical question and has one reason to change | makes placement and review predictable.
- IS-2 | Schema contains declarative semantic facts (WHAT); EDA modules contain reusable computation (HOW) | prevents business metadata from being hard-coded in notebooks/functions.
- IS-3 | Notebook callers use a small public API while private helpers remain module-local | notebooks stay orchestration + interpretation.
- IS-4 | Existing notebook imports continue working during migration | restructuring must not silently break current analysis.
- IS-5 | Acceptance behavior is agreed in Markdown, then executable tests are written before implementation | matches the owner's contract-first workflow.
- IS-6 | No empty phase-mirroring modules or premature target framework are created | avoids architecture that exists only because notebook numbers exist.

- GAP-1 | `inventory.py` currently mixes structural inventory with missing-value and cardinality profiling | boundaries are not yet cohesive | close via inventory/profiling/quality split.
- GAP-2 | `DatasetSpec` has entity/time/required roles but no row-grain metadata | a table-level grain check cannot be contract-driven | close with optional `grain_cols` only.
- GAP-3 | `DAY` is an ordinal integer but temporal code accepts only calendar Date/Datetime/String and converts to months | transaction temporal analysis currently fails and a fabricated calendar could conflict with `WEEK_NO` | close with an ordinal-index path that preserves raw DAY.
- GAP-4 | `report.py` combines inventory, quality, temporal, entity and target | it mirrors a do-everything flow instead of notebook-level orchestration | defer/remove from the public path.
- GAP-5 | `target.py` exists before a target contract has been designed | premature API contradicts the current project stage | keep out of active scope/public workflow.
- GAP-6 | Existing EDA capabilities have no executable behavior tests | a broad rewrite would be unsafe | use Markdown AC approval followed by focused tests per active module.
- GAP-7 | Current `loader.py` is a pandas registry for unrelated datasets while Dunnhumby notebooks/tests call `pl.scan_csv` directly | `COUPON_DISC` can be inferred as Int64 and fail later on decimal values | add a dedicated lazy Polars Dunnhumby scan path with explicit parse overrides.

## Components (topology ledger)

- C1 | Optional row-grain metadata in the existing lightweight schema | active | `src/schema/spec.py`, `src/schema/dunnhumby.py`.
- C2 | Stable LazyFrame loading for Dunnhumby CSVs | active | `src/loader.py`; current transaction direct scan is in `notebooks/01_data_understanding.ipynb:91` and `tests/test_transaction_schema.py:62`.
- C3 | Structural table inventory, key cardinality, and grain summary | active | `src/eda/inventory.py`.
- C4 | Missing, exact duplicate, and constant-column quality reports | active | `src/eda/quality.py`.
- C5 | Entity and ordinal/calendar temporal analysis for eligible tables | active | `src/eda/entity.py`, `src/eda/temporal.py`.
- C6 | Plotting and compatibility imports stay stable while notebook migration and relationship analysis are deferred | active compatibility | `src/eda/visualization/`, `src/eda/descriptive/`.

## Open assumptions (announced defaults)

- Scope default | implement table-level understanding first, using `transaction_data` as the vertical slice, then add specs/runs for one table at a time | explicitly selected by the owner | reversible: yes.
- Schema default | keep `DatasetSpec`/`ColumnRole`; add only optional `grain_cols`; do not add relationship metadata or introduce `TableSpec`/`ColumnSpec` hierarchy now | explicitly selected by the owner and preserves the lightweight schema | reversible: yes.
- Time-axis default | preserve `DAY` as an ordinal day index and never synthesize calendar dates; calculate integer bounds, missing indices, per-day counts, and entity-by-day matrices directly | prevents conflict with `WEEK_NO`/month interpretation | reversible: yes.
- Loader default | add a dedicated Polars LazyFrame Dunnhumby scan API with explicit per-column schema overrides for unstable decimal columns; do not use `infer_schema_length=None` as the routine fix | stable parsing without scanning 141 MB solely for inference | reversible: yes.
- Compatibility default | retain `src/eda/descriptive/*` as thin re-export shims until notebooks are handled separately | notebook changes are explicitly out of current scope | reversible: yes.
- Testing default | contract-first/TDD: update Markdown test plan, get owner approval, write failing tests, then minimal code | follows the owner's stated workflow | reversible: yes.
- Scale default | Polars `LazyFrame`; no new dependency, registry, Pandera/Pydantic schema framework, or caching layer | current stack and data sizes already point to lazy Polars | reversible: yes.

## Findings (cited - path:lines)

- `DatasetSpec` currently exposes only `time_grain`, `role_overrides`, and `required_cols`; no row-grain declaration exists (`src/schema/spec.py:41-50`).
- `TRANSACTION_SPEC` declares required columns and ID overrides only (`src/schema/dunnhumby.py:11-20`).
- `inventory.py` currently emits `n_missing` and `n_unique_approx`, overlapping profiling/quality concerns (`src/eda/inventory.py:17-19`, `src/eda/inventory.py:104-106`).
- `quality.py` already owns missing, duplicate and constant checks (`src/eda/quality.py:13-25`, `src/eda/quality.py:44`).
- `report.py` combines every current EDA section (`src/eda/report.py:28`), while the supplied design explicitly keeps notebook orchestration separate from reusable capabilities.
- `target.py` is implemented (`src/eda/target.py:30`) even though the approved transaction schema still has no target.
- Existing notebook code imports the compatibility facade and calls dataset/quality/temporal/entity separately (`notebooks/01_data_understanding.ipynb:48-52`, `:190`, `:247`, `:329`, `:423`).
- `loader.py` returns pandas DataFrames for an unrelated registry (`src/loader.py:66-81`), while the active EDA stack consumes Polars LazyFrames and the notebook scans transaction directly (`notebooks/01_data_understanding.ipynb:91`).
- `temporal.py` only normalizes Date/Datetime/String (`src/eda/temporal.py:45-56`) and its report is month/calendar-specific (`src/eda/temporal.py:25-42`), so it cannot represent ordinal `DAY` without inventing a calendar.
- `activity_matrix.py` currently performs both temporal computation and plotting and depends on calendar dates (`src/eda/visualization/activity_matrix.py:72`); table-level computation should move to temporal code before a later visualization-only cleanup.
- The only executable tests cover transaction schema behavior; EDA module behavior is not protected (`tests/test_transaction_schema.py`, `tests/test_transaction_schema.md`).

## Decisions (with rationale)

- Keep a functional, responsibility-based module layout; do not create `00_*.py`, `01_*.py`, or empty notebook-mirroring modules.
- Keep `inventory.py`, `quality.py`, `temporal.py`, `entity.py`, and the existing `visualization/` package when each can be described in one sentence.
- Do not create `profiling.py` yet; create it only when approved column-cardinality/profile operations are actually extracted from inventory.
- Defer every relationship declaration and validator to a later multi-table phase.
- Keep private helpers private and expose only notebook-meaningful operations.
- Do not expand or wire `target.py`/`report.py` in this stage; target design remains deferred.
- Preserve current notebook imports through compatibility facades; direct notebook migration is a separate approved task.
- Treat grain as an observation report (`is_unique`, duplicate groups/rows), not a structural exception and never a data mutation.
- Split temporal computation by axis kind: ordinal Int DAY stays numeric; real Date/Datetime remains calendar-based. No epoch/base-date mapping is permitted.
- Keep visualization files stable during the first vertical slice; produce an entity-by-day data result in `temporal.py`; defer changing notebook plotting calls until notebook migration is approved.
- Enforce pure computation: EDA functions accept LazyFrames and return structured reports/DataFrames; only loader performs file I/O and only visualization creates figures.
- Use frozen, slotted dataclasses for report objects and fixed-schema Polars DataFrames for tabular detail; avoid raw result dictionaries.
- Require one small happy test and at least one adverse/boundary test per public function before trusting implementation.
- Keep additions spec-driven and table-agnostic; no `if table == "transaction"` inside EDA computation.
- Preserve public compatibility additively; the package-layout migration from `src` to `src/<package>/` is a separate post-notebook-00 task.

## Scope IN

- Final responsibility map and per-file public/private function ownership.
- Minimal schema contract required for optional table grain.
- A Polars LazyFrame loader path for Dunnhumby with stable transaction numeric parsing.
- `inventory.py`, `quality.py`, `temporal.py`, and `entity.py` behavior for `transaction_data`.
- A staged contract/spec/run checklist for the remaining seven tables; no new module per table.
- Compatibility facade/import migration within Python source.
- Markdown acceptance matrices for transaction inventory/quality/ordinal temporal/entity, followed by focused executable tests, lint/audit and bounded real-data smoke checks.
- Approved acceptance source: `.omo/drafts/notebook-00-test-plan.md`; copy into the repository test documentation only during the execution phase after owner approval.

## Scope OUT (Must NOT have)

- No notebook edits.
- No target creation, target validation, label distribution redesign, propensity window design, feature engineering, or modelling.
- No relationship specification, relationship validation, FK/orphan/join-coverage analysis, or `relationships.py`.
- No `profiling.py` until concrete functions are approved for extraction.
- No seven-table contract values invented without per-table evidence/approval.
- No dtype validation added back to `DatasetSpec`.
- No new framework/dependency, registry, generic plugin system, or package split by arbitrary line count.
- No calendar date synthesized from `DAY`; no assumption that day-index buckets equal calendar months.
- No plotting redesign in the first vertical slice; visualization consumes later-approved computed outputs.

## Open questions

- Resolved A: table-level only; relationships are deferred.
- Resolved B: start with `transaction_data`, then approve one table spec/run at a time.
- Resolved C: `DAY` remains an ordinal integer index; calendar conversion is forbidden.
- Pending approval of the updated brief: contract-first/TDD; Markdown acceptance cases are reviewed before executable tests and code.

## Approval gate
status: awaiting-approval
approach: table-level vertical slice on transaction_data; grain_cols only; stable Polars loader; inventory + quality + entity + ordinal temporal; compatibility preserved; relationships/profiling/target/report/notebooks deferred.
next action after approval: create `.omo/plans/eda-module-responsibilities.md`, run mandatory gap analysis, and write a decision-complete execution plan. Do not implement product code.
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->

"""Customer-by-period activity-count matrix and coverage visualization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from propensity.config import DEFAULT_TIME_FREQ
from propensity.eda._errors import AnalysisColumnError
from propensity.eda._expressions import date_expr

Frame = pl.DataFrame | pl.LazyFrame


@dataclass(frozen=True, slots=True)
class ActivityMatrixInputError(Exception):
    """Raised when activity-matrix input or options are invalid."""

    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class ActivityCoverage:
    """Whole-dataset coverage for the requested time window."""

    n_entities: int
    n_eligible_entities: int
    n_periods: int
    n_active_entity_periods: int
    coverage_pct: float
    sample_size: int


@dataclass(frozen=True, slots=True)
class ActivityMatrixResult:
    """Count matrix, sampled entity coverage, figure, and global coverage."""

    figure: Figure
    matrix: pl.DataFrame
    entity_coverage: pl.DataFrame
    global_coverage: ActivityCoverage


def _month_boundary(value: str | date | None, name: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return date(value.year, value.month, 1)
    normalized = f"{value}-01" if len(value) == 7 else value
    try:
        parsed = date.fromisoformat(normalized)
    except ValueError as error:
        raise ActivityMatrixInputError(
            f"{name} phải có dạng YYYY-MM hoặc YYYY-MM-DD",
        ) from error
    return date(parsed.year, parsed.month, 1)


def _period_label(period: date, freq: str) -> str:
    return period.strftime("%Y-%m") if freq.endswith("mo") else period.isoformat()


def plot_entity_time_matrix(
    df: Frame,
    entity_col: str,
    timestamp_col: str,
    *,
    start: str | date | None = None,
    end: str | date | None = None,
    freq: str = DEFAULT_TIME_FREQ,
    sample_size: int = 100,
    min_active_periods: int = 1,
    unique_col: str | None = None,
    seed: int = 42,
    id_suffix: int = 10,
    annotate: bool = True,
) -> ActivityMatrixResult:
    """Plot per-entity period counts and report coverage for sampled entities."""
    if sample_size <= 0 or min_active_periods <= 0 or id_suffix <= 0:
        raise ActivityMatrixInputError(
            "sample_size, min_active_periods và id_suffix phải lớn hơn 0",
        )
    first_requested = _month_boundary(start, "start")
    last_requested = _month_boundary(end, "end")
    if (
        first_requested is not None
        and last_requested is not None
        and first_requested > last_requested
    ):
        raise ActivityMatrixInputError("start phải nhỏ hơn hoặc bằng end")

    lazy = df.lazy() if isinstance(df, pl.DataFrame) else df
    schema = lazy.collect_schema()
    required = [entity_col, timestamp_col]
    if unique_col is not None:
        required.append(unique_col)
    for column in required:
        if column not in schema:
            raise AnalysisColumnError(column, "không tồn tại trong dữ liệu")

    selection = [
        pl.col(entity_col),
        date_expr(timestamp_col, schema[timestamp_col]).alias("__date"),
    ]
    if unique_col is not None:
        selection.append(pl.col(unique_col).alias("__unique_value"))
    normalized = (
        lazy.select(selection)
        .filter(pl.col(entity_col).is_not_null() & pl.col("__date").is_not_null())
        .with_columns(pl.col("__date").dt.truncate(freq).alias("period"))
    )
    if first_requested is not None:
        normalized = normalized.filter(pl.col("period") >= first_requested)
    if last_requested is not None:
        normalized = normalized.filter(pl.col("period") <= last_requested)

    count_expression = (
        pl.col("__unique_value").drop_nulls().n_unique()
        if unique_col is not None
        else pl.len()
    )
    activity = (
        normalized.group_by(entity_col, "period")
        .agg(count_expression.alias("n_records"))
        .filter(pl.col("n_records") > 0)
    )
    bounds = (
        activity.select(
            pl.col("period").min().alias("first_period"),
            pl.col("period").max().alias("last_period"),
        )
        .collect()
        .row(0, named=True)
    )
    if bounds["first_period"] is None:
        raise ActivityMatrixInputError("Không có entity nào với thời gian hợp lệ để vẽ")

    first_period = first_requested or bounds["first_period"]
    last_period = last_requested or bounds["last_period"]
    periods = pl.date_range(first_period, last_period, interval=freq, eager=True)
    n_periods = periods.len()
    entity_stats = activity.group_by(entity_col).agg(
        pl.len().alias("active_periods"),
        pl.col("n_records").sum().alias("total_records"),
    )
    global_stats = (
        entity_stats.select(
            pl.len().alias("n_entities"),
            (pl.col("active_periods") >= min_active_periods)
            .sum()
            .alias("n_eligible_entities"),
            pl.col("active_periods").sum().alias("n_active_pairs"),
        )
        .collect()
        .row(0, named=True)
    )
    n_entities = int(global_stats["n_entities"])

    selected = (
        entity_stats.filter(pl.col("active_periods") >= min_active_periods)
        .with_columns(pl.col(entity_col).hash(seed=seed).alias("__sample_key"))
        .sort("__sample_key")
        .head(sample_size)
        .select(entity_col)
        .collect()
        .with_row_index("__row")
    )
    if selected.is_empty():
        raise ActivityMatrixInputError("Không có entity nào đạt min_active_periods")
    selected_values = selected[entity_col].to_list()
    sampled_activity = activity.filter(
        pl.col(entity_col).is_in(selected_values)
    ).collect()
    grid = (
        selected.join(pl.DataFrame({"period": periods}), how="cross")
        .join(sampled_activity, on=[entity_col, "period"], how="left")
        .with_columns(pl.col("n_records").fill_null(0))
        .sort("__row", "period")
    )
    matrix_values = np.asarray(grid["n_records"].to_list()).reshape(
        selected.height, n_periods
    )
    matrix_data: dict[str, pl.Series | list[str]] = {entity_col: selected[entity_col]}
    for index, period in enumerate(periods):
        matrix_data[_period_label(period, freq)] = pl.Series(matrix_values[:, index])
    matrix = pl.DataFrame(matrix_data)
    entity_coverage = (
        selected.drop("__row")
        .join(
            entity_stats.filter(pl.col(entity_col).is_in(selected_values)).collect(),
            on=entity_col,
        )
        .with_columns(
            pl.lit(n_periods).alias("total_periods"),
            (100.0 * pl.col("active_periods") / n_periods).alias("coverage_pct"),
        )
        .select(
            entity_col,
            "active_periods",
            "total_periods",
            "coverage_pct",
            "total_records",
        )
    )

    figure, axis = plt.subplots(
        figsize=(
            min(24, max(9, 0.5 * n_periods + 3)),
            min(30, max(5, 0.5 * selected.height + 2)),
        ),
        constrained_layout=True,
    )
    maximum = max(1, int(matrix_values.max()))
    image = axis.imshow(
        matrix_values,
        aspect="auto",
        cmap=LinearSegmentedColormap.from_list("activity", ["#FFFFFF", "#00B050"]),
        vmin=0,
        vmax=maximum,
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(-0.5, n_periods, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, selected.height, 1), minor=True)
    axis.grid(which="minor", color="#D9D9D9", linewidth=0.5)
    axis.tick_params(which="minor", length=0)
    axis.set_xticks(
        range(n_periods),
        [_period_label(period, freq) for period in periods],
        rotation=90,
        fontsize=8,
    )
    entity_labels = [str(value)[-id_suffix:] for value in selected_values]
    axis.set_yticks(range(selected.height), entity_labels, fontsize=8)
    if annotate:
        for row in range(matrix_values.shape[0]):
            for column in range(matrix_values.shape[1]):
                value = int(matrix_values[row, column])
                if value > 0:
                    axis.text(
                        column,
                        row,
                        value,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if value > 0.6 * maximum else "black",
                    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.01)
    colorbar.set_label("Số bản ghi trong kỳ")
    colorbar.locator = MaxNLocator(integer=True)
    colorbar.update_ticks()
    axis.set(
        title=f"Hoạt động theo thời gian — mẫu {selected.height:,}/{n_entities:,} entities",
        xlabel="Kỳ hoạt động",
        ylabel=f"{entity_col} ({id_suffix} ký tự cuối)",
    )

    n_active_pairs = int(global_stats["n_active_pairs"])
    return ActivityMatrixResult(
        figure=figure,
        matrix=matrix,
        entity_coverage=entity_coverage,
        global_coverage=ActivityCoverage(
            n_entities=n_entities,
            n_eligible_entities=int(global_stats["n_eligible_entities"]),
            n_periods=n_periods,
            n_active_entity_periods=n_active_pairs,
            coverage_pct=100.0 * n_active_pairs / (n_entities * n_periods),
            sample_size=selected.height,
        ),
    )

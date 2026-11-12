from __future__ import annotations

from typing import Final

import matplotlib
import polars as pl
from matplotlib.figure import Figure

from propensity.eda.temporal_behavior_shift import build_household_behavior_shift
from propensity.eda.transaction_models import TransactionProfileConfig
from propensity.eda.visualization import plot_household_behavior_shift

matplotlib.use("Agg")

TEST_CONFIG: Final = TransactionProfileConfig(
    entity_col="household_key",
    basket_col="BASKET_ID",
    time_col="DAY",
    period_col="WEEK_NO",
    product_col="PRODUCT_ID",
    store_col="STORE_ID",
    quantity_col="QUANTITY",
    sales_col="SALES_VALUE",
    discount_cols=("RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC"),
)


def _transactions() -> pl.LazyFrame:
    rows: list[tuple[int, int, int, int]] = []
    for household in range(1, 7):
        for day in (10, 30, 50, 70, 90, 100):
            rows.append((household, household * 1000 + day, day, household + day))
        recent_days = (125, 130, 135) if household <= 3 else ()
        for day in recent_days:
            rows.append((household, household * 1000 + day, day, household + day))
    row_count = len(rows)
    return pl.LazyFrame(
        {
            "household_key": [row[0] for row in rows],
            "BASKET_ID": [row[1] for row in rows],
            "DAY": [row[2] for row in rows],
            "WEEK_NO": [(row[2] - 1) // 7 + 1 for row in rows],
            "PRODUCT_ID": [row[3] for row in rows],
            "STORE_ID": [7] * row_count,
            "QUANTITY": [1] * row_count,
            "SALES_VALUE": [1.0] * row_count,
            "RETAIL_DISC": [0.0] * row_count,
            "COUPON_DISC": [0.0] * row_count,
            "COUPON_MATCH_DISC": [0.0] * row_count,
        }
    )


def test_shift_keeps_dormant_households_and_uses_non_overlapping_periods() -> None:
    # Given / When
    result = build_household_behavior_shift(
        _transactions(),
        TEST_CONFIG,
        recent_days=30,
        history_days=90,
        cutoff_day=135,
    )

    # Then
    assert result.height == 6
    assert result["recent_start_day"][0] == 106
    assert result["history_end_day"][0] == 105
    assert result.filter(pl.col("frequency_state") == "Dormant").height == 3


def test_behavior_shift_figure_has_population_groups_and_slopegraph() -> None:
    # Given
    households = build_household_behavior_shift(
        _transactions(),
        TEST_CONFIG,
        recent_days=30,
        history_days=90,
        cutoff_day=135,
    )

    # When
    figure: Figure = plot_household_behavior_shift(households)

    # Then
    assert len(figure.axes) == 3

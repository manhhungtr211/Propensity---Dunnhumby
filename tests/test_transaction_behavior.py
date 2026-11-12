from __future__ import annotations

from typing import Final

import matplotlib
import polars as pl
import pytest
from matplotlib.figure import Figure

from propensity.eda.transaction_behavior import (
    TransactionBehaviorTables,
    build_transaction_behavior_tables,
)
from propensity.eda.transaction_models import TransactionProfileConfig
from propensity.eda.visualization import (
    plot_basket_size_value_structure,
    plot_behavioral_patterns,
    plot_cross_behavior,
    plot_household_spending,
    plot_purchase_frequency,
    plot_store_usage,
    plot_transaction_discount_usage,
)

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


@pytest.fixture
def behavior_tables() -> TransactionBehaviorTables:
    transactions = pl.LazyFrame(
        {
            "household_key": [1, 1, 1, 2, 2, 3, 3, 4, 4],
            "BASKET_ID": [100, 100, 101, 200, 201, 300, 301, 400, 401],
            "DAY": [1, 1, 2, 1, 3, 1, 4, 2, 5],
            "WEEK_NO": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "PRODUCT_ID": [10, 11, 10, 11, 12, 10, 13, 12, 13],
            "STORE_ID": [7, 7, 7, 8, 9, 7, 7, 8, 8],
            "QUANTITY": [99, 1, 1, 1, 1, 1, 1, 1, 1],
            "SALES_VALUE": [5.0, 3.0, 0.0, 8.0, 12.0, 2.0, 4.0, 20.0, 10.0],
            "RETAIL_DISC": [-1.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, -3.0, 0.0],
            "COUPON_DISC": [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0],
            "COUPON_MATCH_DISC": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        }
    )
    products = pl.LazyFrame(
        {
            "PRODUCT_ID": [10, 11, 12, 13],
            "DEPARTMENT": ["GROCERY", "GROCERY", "DRUG GM", "PRODUCE"],
        }
    )
    return build_transaction_behavior_tables(transactions, products, TEST_CONFIG)


def test_behavior_tables_apply_agreed_spending_and_grain(
    behavior_tables: TransactionBehaviorTables,
) -> None:
    # Given / When
    basket = behavior_tables.baskets.filter(pl.col("BASKET_ID") == 100).row(
        0, named=True
    )
    household = behavior_tables.households.filter(pl.col("household_key") == 1).row(
        0, named=True
    )

    # Then
    assert basket["customer_spend"] == pytest.approx(7.0)
    assert basket["product_count"] == 2
    assert basket["category_count"] == 1
    assert household["basket_count"] == 2
    assert household["valid_value_basket_count"] == 1
    assert household["total_spend"] == pytest.approx(7.0)


def test_behavior_tables_ignore_quantity_for_basket_size(
    behavior_tables: TransactionBehaviorTables,
) -> None:
    # Given / When
    basket = behavior_tables.baskets.filter(pl.col("BASKET_ID") == 100).row(
        0, named=True
    )

    # Then
    assert basket["product_count"] == 2


def test_category_penetration_counts_each_basket_once_per_department(
    behavior_tables: TransactionBehaviorTables,
) -> None:
    # Given / When
    result = behavior_tables.category_penetration

    # Then
    shares = dict(zip(result["category"], result["basket_share"], strict=True))
    assert shares == pytest.approx({"GROCERY": 0.5, "DRUG GM": 0.25, "PRODUCE": 0.25})


def test_category_penetration_groups_low_share_departments_by_basket_union() -> None:
    # Given
    transactions = pl.LazyFrame(
        {
            "household_key": [*range(1, 12), 11],
            "BASKET_ID": [*range(100, 111), 110],
            "DAY": [1] * 12,
            "WEEK_NO": [1] * 12,
            "PRODUCT_ID": [*([10] * 10), 20, 30],
            "STORE_ID": [7] * 12,
            "QUANTITY": [1] * 12,
            "SALES_VALUE": [1.0] * 12,
            "RETAIL_DISC": [0.0] * 12,
            "COUPON_DISC": [0.0] * 12,
            "COUPON_MATCH_DISC": [0.0] * 12,
        }
    )
    products = pl.LazyFrame(
        {
            "PRODUCT_ID": [10, 20, 30],
            "DEPARTMENT": ["GROCERY", "FLORAL", "SEAFOOD"],
        }
    )

    # When
    result = build_transaction_behavior_tables(
        transactions,
        products,
        TEST_CONFIG,
    ).category_penetration

    # Then
    shares = dict(zip(result["category"], result["basket_share"], strict=True))
    assert shares == pytest.approx({"GROCERY": 10 / 11, "Other": 1 / 11})


def test_notebook_03_plot_functions_return_expected_panels(
    behavior_tables: TransactionBehaviorTables,
) -> None:
    # Given / When
    figures: tuple[Figure, ...] = (
        plot_purchase_frequency(behavior_tables.households),
        plot_basket_size_value_structure(
            behavior_tables.baskets,
            behavior_tables.category_penetration,
        ),
        plot_household_spending(behavior_tables.households),
        plot_store_usage(behavior_tables.households),
        plot_transaction_discount_usage(
            behavior_tables.baskets,
            behavior_tables.households,
        ),
        plot_cross_behavior(behavior_tables.households),
        plot_behavioral_patterns(behavior_tables.households),
    )

    # Then
    assert [len(figure.axes) for figure in figures] == [2, 4, 3, 3, 4, 2, 2]

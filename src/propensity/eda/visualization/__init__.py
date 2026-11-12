"""Visualization helpers for exploratory data analysis."""

from propensity.eda.visualization.activity_matrix import (
    ActivityCoverage,
    ActivityMatrixResult,
    plot_entity_time_matrix,
)
from propensity.eda.visualization.plots import (
    plot_active_periods_per_entity,
    plot_missing_values,
    plot_records_over_time,
    plot_records_per_entity,
)
from propensity.eda.visualization.transaction import plot_return_interval_distribution
from propensity.eda.visualization.transaction_behavior_overview import (
    plot_basket_size_value_structure,
    plot_household_spending,
    plot_purchase_frequency,
)
from propensity.eda.visualization.transaction_behavior_usage import (
    plot_behavioral_patterns,
    plot_cross_behavior,
    plot_store_usage,
    plot_transaction_discount_usage,
)

__all__ = [
    "ActivityCoverage",
    "ActivityMatrixResult",
    "plot_active_periods_per_entity",
    "plot_basket_size_value_structure",
    "plot_behavioral_patterns",
    "plot_cross_behavior",
    "plot_entity_time_matrix",
    "plot_household_spending",
    "plot_missing_values",
    "plot_purchase_frequency",
    "plot_records_over_time",
    "plot_records_per_entity",
    "plot_return_interval_distribution",
    "plot_store_usage",
    "plot_transaction_discount_usage",
]

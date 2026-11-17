from __future__ import annotations

from itertools import pairwise

import pandas as pd
import pytest

from propensity.labels import (
    DEFAULT_PURCHASE_LABEL_SPEC,
    PurchaseLabelError,
    PurchaseLabelSpec,
    build_purchase_labels,
)


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "household_key": [1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
            "DAY": [50, 100, 107, 80, 107, 114, 108, 115, 80, 114],
            "QUANTITY": [1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
            "SALES_VALUE": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0],
        }
    ).astype({"household_key": "Int64", "DAY": "Int64"})


def _spec() -> PurchaseLabelSpec:
    return PurchaseLabelSpec(
        observation_days=10,
        horizon_days=7,
        first_cutoff_day=100,
        last_cutoff_day=107,
        cutoff_step_days=7,
    )


def test_default_cutoff_grid_runs_from_140_to_700_by_7() -> None:
    # Given / When
    cutoffs = DEFAULT_PURCHASE_LABEL_SPEC.cutoff_days

    # Then
    assert cutoffs[0] == 140
    assert cutoffs[-1] == 700
    assert len(cutoffs) == 81
    assert all(right - left == 7 for left, right in pairwise(cutoffs))


def test_label_uses_cutoff_as_first_target_day() -> None:
    # Given
    transactions = _transactions()

    # When
    result = build_purchase_labels(transactions, _spec())

    # Then
    cutoff_100 = result.loc[result["cutoff_day"] == 100]
    assert cutoff_100[["household_key", "label"]].values.tolist() == [
        [1, 1],
        [2, 0],
        [4, 0],
    ]
    assert cutoff_100["target_start_day"].iat[0] == 100
    assert cutoff_100["target_end_day"].iat[0] == 106


def test_panel_keeps_known_households_and_marks_full_history() -> None:
    # Given / When
    result = build_purchase_labels(_transactions(), _spec())

    # Then
    cutoff_100 = result.loc[result["cutoff_day"] == 100]
    assert 3 not in cutoff_100["household_key"].tolist()
    household_1 = cutoff_100.loc[cutoff_100["household_key"] == 1].iloc[0]
    assert bool(household_1["has_full_observation_window"])
    assert household_1["observation_start_day"] == 90
    assert household_1["observation_end_day"] == 99


def test_household_first_seen_on_cutoff_is_not_eligible() -> None:
    # Given
    transactions = _transactions()
    transactions.loc[len(transactions)] = [5, 100, 1, 1.0]

    # When
    result = build_purchase_labels(transactions, _spec())

    # Then
    cutoff_100 = result.loc[result["cutoff_day"] == 100]
    assert 5 not in cutoff_100["household_key"].tolist()


def test_invalid_purchase_does_not_create_positive_label() -> None:
    # Given / When
    result = build_purchase_labels(_transactions(), _spec())

    # Then
    cutoff_107 = result.loc[result["cutoff_day"] == 107]
    household_4 = cutoff_107.loc[cutoff_107["household_key"] == 4].iloc[0]
    assert household_4["label"] == 0


def test_build_purchase_labels_does_not_mutate_input() -> None:
    # Given
    transactions = _transactions()
    before = transactions.copy(deep=True)

    # When
    build_purchase_labels(transactions, _spec())

    # Then
    pd.testing.assert_frame_equal(transactions, before)


def test_incomplete_final_horizon_fails_clearly() -> None:
    # Given
    incomplete = _transactions().loc[lambda frame: frame["DAY"] <= 108]

    # When / Then
    with pytest.raises(PurchaseLabelError, match="Target window cuối"):
        build_purchase_labels(incomplete, _spec())

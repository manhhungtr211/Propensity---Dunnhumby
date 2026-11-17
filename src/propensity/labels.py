"""Leakage-safe household purchase labels across temporal cutoffs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd


class PurchaseLabelError(ValueError):
    """Raised when purchase-label configuration or input is invalid."""


@dataclass(frozen=True, slots=True)
class PurchaseLabelSpec:
    """Observation, prediction-horizon, and target-start policy.

    A cutoff is the first day of its target window. The observation window ends
    one day before the cutoff, so the two windows never overlap.
    """

    observation_days: int = 90
    horizon_days: int = 7
    first_cutoff_day: int = 140
    last_cutoff_day: int = 700
    cutoff_step_days: int = 7

    def __post_init__(self) -> None:
        positive_values = (
            self.observation_days,
            self.horizon_days,
            self.first_cutoff_day,
            self.last_cutoff_day,
            self.cutoff_step_days,
        )
        if any(value <= 0 for value in positive_values):
            raise PurchaseLabelError("Các tham số thời gian phải lớn hơn 0")
        if self.first_cutoff_day < self.observation_days:
            raise PurchaseLabelError("Cutoff đầu tiên không đủ observation window")
        if self.last_cutoff_day < self.first_cutoff_day:
            raise PurchaseLabelError("Cutoff cuối phải từ cutoff đầu tiên trở đi")
        if (self.last_cutoff_day - self.first_cutoff_day) % self.cutoff_step_days:
            raise PurchaseLabelError("Cutoff cuối không nằm trên lưới cutoff")

    @property
    def cutoff_days(self) -> tuple[int, ...]:
        """Return the inclusive, evenly spaced cutoff grid."""
        return tuple(
            range(
                self.first_cutoff_day,
                self.last_cutoff_day + 1,
                self.cutoff_step_days,
            )
        )


DEFAULT_PURCHASE_LABEL_SPEC: Final = PurchaseLabelSpec()
_REQUIRED_COLUMNS: Final = (
    "household_key",
    "DAY",
    "QUANTITY",
    "SALES_VALUE",
)


def build_purchase_labels(
    transactions: pd.DataFrame,
    spec: PurchaseLabelSpec = DEFAULT_PURCHASE_LABEL_SPEC,
) -> pd.DataFrame:
    """Create one any-product purchase label per known household and cutoff."""
    missing = [
        column for column in _REQUIRED_COLUMNS if column not in transactions.columns
    ]
    if missing:
        raise PurchaseLabelError(f"Thiếu cột giao dịch bắt buộc: {missing}")

    valid_events = (
        transactions.loc[
            (transactions["QUANTITY"] > 0) & (transactions["SALES_VALUE"] >= 0),
            ["household_key", "DAY"],
        ]
        .dropna()
        .drop_duplicates()
    )
    if valid_events.empty:
        raise PurchaseLabelError("Không có purchase hợp lệ để tạo label")

    max_day = int(valid_events["DAY"].max())
    if spec.last_cutoff_day + spec.horizon_days - 1 > max_day:
        raise PurchaseLabelError(
            "Target window cuối vượt quá thời gian quan sát của dữ liệu"
        )

    first_purchases = (
        valid_events.groupby("household_key", as_index=False, sort=True)["DAY"]
        .min()
        .rename(columns={"DAY": "first_purchase_day"})
    )
    panels: list[pd.DataFrame] = []
    for cutoff_day in spec.cutoff_days:
        observation_start_day = cutoff_day - spec.observation_days
        observation_end_day = cutoff_day - 1
        target_start_day = cutoff_day
        target_end_day = cutoff_day + spec.horizon_days - 1
        eligible = first_purchases.loc[
            first_purchases["first_purchase_day"] <= observation_end_day
        ].copy()
        positive_households = valid_events.loc[
            valid_events["DAY"].between(target_start_day, target_end_day),
            "household_key",
        ].drop_duplicates()
        eligible["label"] = (
            eligible["household_key"].isin(positive_households).astype("UInt8")
        )
        eligible["cutoff_day"] = cutoff_day
        eligible["observation_start_day"] = observation_start_day
        eligible["observation_end_day"] = observation_end_day
        eligible["target_start_day"] = target_start_day
        eligible["target_end_day"] = target_end_day
        eligible["observation_days"] = spec.observation_days
        eligible["target_horizon_days"] = spec.horizon_days
        eligible["cutoff_step_days"] = spec.cutoff_step_days
        eligible["exposure_days_at_cutoff"] = (
            cutoff_day - eligible["first_purchase_day"]
        )
        eligible["has_full_observation_window"] = (
            eligible["exposure_days_at_cutoff"] >= spec.observation_days
        )
        panels.append(eligible)

    result = pd.concat(panels, ignore_index=True)
    integer_columns = [
        "household_key",
        "first_purchase_day",
        "cutoff_day",
        "observation_start_day",
        "observation_end_day",
        "target_start_day",
        "target_end_day",
        "observation_days",
        "target_horizon_days",
        "cutoff_step_days",
        "exposure_days_at_cutoff",
    ]
    result[integer_columns] = result[integer_columns].astype("Int64")
    return result.sort_values(["cutoff_day", "household_key"], ignore_index=True)

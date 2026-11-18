"""Department-group definitions and label-share preparation."""

from __future__ import annotations

from typing import Final

import pandas as pd

RARE_DEPARTMENT_GROUP: Final = "Hiếm (n < 100)"
OTHER_DEPARTMENT_GROUP: Final = "Ngành hàng khác"
DEPARTMENT_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "Nhu yếu phẩm thiết yếu & Tươi sống (Core Staples)": (
        "GROCERY",
        "PRODUCE",
        "MEAT",
        "DELI",
        "PASTRY",
    ),
    "Dược phẩm & Chăm sóc cá nhân (Health & Beauty)": (
        "DRUG GM",
        "NUTRITION",
        "HBC",
        "COSMETICS",
        "RX",
    ),
    "Tiện ích & Thực phẩm đóng gói/đông lạnh (Routine & Convenience)": (
        "PHOTO",
        "KIOSK-GAS",
        "DAIRY DELI",
        "FROZEN",
        "POSTAL",
    ),
    "Ăn uống tại chỗ & Chế biến sẵn (In-Store Dining)": (
        "CHEF SHOPPE",
        "DELI/SNACK BAR",
        "RESTAURANT",
    ),
    "Ngành hàng chuyên biệt & Sở thích (Specialty & Lifestyle)": (
        "FLORAL",
        "GARDEN CENTER",
    ),
}
DEPARTMENT_TO_GROUP: Final[dict[str, str]] = {
    department: group
    for group, departments in DEPARTMENT_GROUPS.items()
    for department in departments
}


def build_department_group_shares(summary: pd.DataFrame) -> pd.DataFrame:
    """Average department-level label shares within the requested groups."""
    department_data = summary.loc[summary["dimension"] == "department"]
    counts = department_data.pivot_table(
        index="category",
        columns="label",
        values="n_observations",
        aggfunc="sum",
        fill_value=0,
    ).reindex(columns=[0, 1], fill_value=0)
    totals = counts.sum(axis=1)
    department_shares = counts.div(totals, axis=0).rename(
        columns={0: "Negative", 1: "Positive"}
    )
    department_shares["group"] = (
        department_shares.index.to_series()
        .map(DEPARTMENT_TO_GROUP)
        .fillna(OTHER_DEPARTMENT_GROUP)
    )
    department_shares.loc[totals < 100, "group"] = RARE_DEPARTMENT_GROUP
    group_shares = department_shares.groupby("group")[["Negative", "Positive"]].mean()
    return group_shares.sort_values("Negative", ascending=False)

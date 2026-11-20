"""Targeted script to generate S9_raw_vs_rf with unbuffered progress."""

from __future__ import annotations

import time
import matplotlib.pyplot as plt
import pandas as pd

from propensity.eda.customer_label_relation import (
    CustomerLabelConfig,
    build_customer_cutoff_panel,
)
from propensity.eda.propensity_rates import (
    add_verdicts,
    excess_over_rf,
    feature_signal_summary,
    join_features,
    quantile_bin,
    rf_baseline,
)
from propensity.eda.slide_figures import fig_v2
from propensity.features.basket import basket_features, basket_table, day_basket_table
from propensity.features.category import (
    CategoryConfig,
    category_features,
    commodity_stats,
    define_staples,
    household_day_category,
    merch_lines,
)
from propensity.features.demographic import (
    demographic_features,
    group_demographic,
)
from propensity.features.promotion import (
    campaign_features,
    promo_features,
)
from propensity.features.temporal import (
    lapse_bin,
    temporal_features,
    trend_bin,
)
from propensity.viz.slide_style import save_slide


def main() -> None:
    t0 = time.time()
    print("1. Loading raw tables...", flush=True)
    tx = pd.read_csv("data/raw/Dunnhumby/transaction_data.csv")
    prod = pd.read_csv("data/raw/Dunnhumby/product.csv")
    demo = pd.read_csv("data/raw/Dunnhumby/hh_demographic.csv")
    camp = pd.read_csv("data/raw/Dunnhumby/campaign_table.csv")
    camp_desc = pd.read_csv("data/raw/Dunnhumby/campaign_desc.csv")
    coup = pd.read_csv("data/raw/Dunnhumby/coupon_redempt.csv")

    cat_cfg = CategoryConfig()
    lines = merch_lines(tx, prod, cat_cfg)

    print("2. Building customer cutoff panel...", flush=True)
    panel = build_customer_cutoff_panel(tx)
    base_rf = rf_baseline(panel)
    panel["expected_rf"] = base_rf.panel["expected_rf"]
    cutoffs = panel["cutoff_day"].unique().tolist()
    print(f"Panel rows: {len(panel)}, cutoffs: {len(cutoffs)}", flush=True)

    print("3. Computing temporal features...", flush=True)
    pdays_distinct = tx[tx["SALES_VALUE"] > 0][["household_key", "DAY"]].drop_duplicates()
    tf = temporal_features(pdays_distinct, cutoffs)
    tf["trend_bin"] = trend_bin(tf["recent_ipt_ratio_c"])
    panel = join_features(panel, tf[["household_key", "cutoff_day", "trend_bin"]])

    print("4. Computing demographic features...", flush=True)
    demo_grp = group_demographic(demo)
    panel = demographic_features(panel, demo_grp)

    print("5. Computing campaign features...", flush=True)
    cf = campaign_features(camp, camp_desc, coup, panel)
    panel = join_features(panel, cf)

    print("6. Computing category features...", flush=True)
    stats = commodity_stats(lines, cat_cfg)
    staples = define_staples(stats, cat_cfg)
    staple_col = "commodity" if "commodity" in staples.columns else "COMMODITY_DESC"
    staple_set = set(staples[staples["is_staple"]][staple_col].tolist())
    hdc = household_day_category(lines, staple_set)
    cat_feats = category_features(hdc, cutoffs)
    panel = join_features(panel, cat_feats)

    print("7. Computing promo features...", flush=True)
    p_feats = promo_features(lines, cutoffs)
    panel = join_features(panel, p_feats)

    print("8. Computing basket features...", flush=True)
    baskets = basket_table(lines)
    day_tbl = day_basket_table(baskets, lines=lines)
    bf = basket_features(day_tbl, cutoffs)
    panel = join_features(panel, bf)

    print("9. Creating quantile bins for signal summary...", flush=True)
    panel["basket_size_bin"] = quantile_bin(panel["median_products_per_day_c"], q=5, prefix="Q")
    panel["last_basket_bin"] = quantile_bin(panel["last_basket_ratio_c"], q=5, prefix="Q")
    if "spend_bin" not in panel.columns:
        panel["spend_bin"] = quantile_bin(panel["spend_per_trip"], q=5, prefix="Q")
    panel["promo_share_bin"] = quantile_bin(panel["promo_spend_share_c"], q=5, prefix="Q")
    panel["discount_depth_bin"] = quantile_bin(panel["discount_depth_c"], q=5, prefix="Q")
    panel["staple_bin"] = quantile_bin(panel["staple_spend_share_c"], q=5, prefix="Q")
    panel["entropy_bin"] = quantile_bin(panel["dept_entropy_c"], q=5, prefix="Q")
    panel["private_bin"] = quantile_bin(panel["private_label_share_c"], q=5, prefix="Q")
    panel["fuel_bin"] = panel["buys_fuel_c"]
    panel["lapse_bin"] = lapse_bin(panel["lapse_ratio_c"])

    print("10. Computing excess tables and verdicts...", flush=True)
    results = {
        "Xu hướng nhịp mua": add_verdicts(excess_over_rf(panel, "trend_bin")),
        "Band IPT": add_verdicts(excess_over_rf(panel, "ipt_band_c")),
        "Lapse ratio": add_verdicts(excess_over_rf(panel, "lapse_bin")),
        "Đang trong campaign": add_verdicts(excess_over_rf(panel, "active_campaign_type_c")),
        "Đã redeem coupon": add_verdicts(excess_over_rf(panel, "redeemed_coupon_c")),
        "Có demographic": add_verdicts(excess_over_rf(panel, "has_demographic")),
        "Spend/trip": add_verdicts(excess_over_rf(panel, "spend_bin")),
        "Kích thước giỏ": add_verdicts(excess_over_rf(panel, "basket_size_bin")),
        "Giỏ gần nhất": add_verdicts(excess_over_rf(panel, "last_basket_bin")),
        "Tỷ trọng mua giảm giá": add_verdicts(excess_over_rf(panel, "promo_share_bin")),
        "Độ sâu giảm giá": add_verdicts(excess_over_rf(panel, "discount_depth_bin")),
        "Đã dùng coupon": add_verdicts(excess_over_rf(panel, "used_coupon_c")),
        "Đổ xăng": add_verdicts(excess_over_rf(panel, "fuel_bin")),
        "Đa dạng department": add_verdicts(excess_over_rf(panel, "entropy_bin")),
        "Tỷ trọng hàng thiết yếu": add_verdicts(excess_over_rf(panel, "staple_bin")),
        "Private label": add_verdicts(excess_over_rf(panel, "private_bin")),
    }

    summary_df = feature_signal_summary(results)
    print("Feature Signal Summary:")
    print(summary_df.to_string())

    fig = fig_v2(summary_df)
    png_path, svg_path = save_slide(fig, "S9_raw_vs_rf")
    plt.close(fig)
    print(f"Saved: {png_path} and {svg_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

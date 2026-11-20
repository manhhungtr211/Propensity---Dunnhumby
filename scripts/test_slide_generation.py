"""End-to-end test script to verify slide figures generation and match against expectations."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from propensity.eda.customer_label_relation import (
    CustomerLabelConfig,
    build_customer_cutoff_panel,
)
from propensity.eda.demographic_plots import plot_demographic_coverage
from propensity.eda.propensity_rates import (
    excess_over_rf,
    add_verdicts,
    rf_baseline,
    feature_signal_summary,
)
from propensity.eda.promotion import campaign_reach_by_rf, weekly_promo_summary
from propensity.eda.slide_figures import (
    fig_c1,
    fig_c2,
    fig_d1,
    fig_l1,
    fig_l2a,
    fig_l2b,
    fig_l3,
    fig_p1,
    fig_p2,
    fig_q1,
    fig_v1,
    fig_v2,
    fig_v3,
    ipt_gap_distribution,
    table_coverage,
)
from propensity.features.basket import basket_size_bands, basket_table
from propensity.features.category import CategoryConfig, department_summary, merch_lines
from propensity.features.demographic import demographic_coverage_profile
from propensity.features.temporal import cutoff_return_rates, weekly_activity
from propensity.viz.slide_style import save_slide, set_slide_style


def main() -> None:
    print("1. Loading raw data...")
    tx = pd.read_csv("data/raw/Dunnhumby/transaction_data.csv")
    demo = pd.read_csv("data/raw/Dunnhumby/hh_demographic.csv")
    camp = pd.read_csv("data/raw/Dunnhumby/campaign_table.csv")
    coup = pd.read_csv("data/raw/Dunnhumby/coupon_redempt.csv")
    prod = pd.read_csv("data/raw/Dunnhumby/product.csv")

    cat_cfg = CategoryConfig()
    lines = merch_lines(tx, prod, cat_cfg)

    print("2. Testing table_coverage (D1)...")
    cov = table_coverage(tx, camp, coup, demo)
    print("Table coverage:")
    print(cov)
    fig_d1_out = fig_d1(cov)
    p_png, p_svg = save_slide(fig_d1_out, "S2_table_coverage")
    print(f"Saved: {p_png}")
    plt.close(fig_d1_out)

    print("3. Testing IPT gap distribution (L2a, L2b)...")
    df_valid = tx[tx["SALES_VALUE"] > 0][["household_key", "DAY"]].drop_duplicates().sort_values(["household_key", "DAY"])
    gaps = ipt_gap_distribution(df_valid)
    print(f"Total gaps: {len(gaps)}, median: {np.median(gaps):.1f}")
    print(f"<=3d: {(gaps<=3).mean()*100:.1f}%, <=7d: {(gaps<=7).mean()*100:.1f}%, <=14d: {(gaps<=14).mean()*100:.1f}%")

    fig_l2a_out = fig_l2a(gaps)
    save_slide(fig_l2a_out, "S5_ipt_hist")
    plt.close(fig_l2a_out)

    fig_l2b_out = fig_l2b(gaps)
    save_slide(fig_l2b_out, "S5_ipt_cumulative")
    plt.close(fig_l2b_out)

    print("4. Testing weekly activity (L1)...")
    weekly = weekly_activity(lines)
    post_140_mean = weekly[weekly["min_day"] >= 140]["active_hh"].mean()
    print(f"Weekly active post-140 mean: {post_140_mean:.1f}")
    fig_l1_out = fig_l1(weekly)
    save_slide(fig_l1_out, "S4_weekly_active")
    plt.close(fig_l1_out)

    print("5. Testing basket bands (C1) and department pareto (C2)...")
    baskets = basket_table(lines)
    bands = basket_size_bands(baskets)
    print("Bands:")
    print(bands[["band", "pct_baskets", "pct_spend", "median_spend"]])
    fig_c1_out = fig_c1(bands)
    save_slide(fig_c1_out, "S11_basket_bands")
    plt.close(fig_c1_out)

    depts = department_summary(lines, cat_cfg)
    dept_col = "department" if "department" in depts.columns else "DEPARTMENT"
    print("Top depts:")
    print(depts.head(5)[[dept_col, "spend_share", "cum_spend_share"]])
    fig_c2_out = fig_c2(depts)
    save_slide(fig_c2_out, "S11_department_pareto")
    plt.close(fig_c2_out)

    print("6. Testing weekly promo (P1)...")
    wp = weekly_promo_summary(lines)
    print(f"Weekly promo mean: {wp['promo_spend_share'].mean()*100:.2f}%")
    fig_p1_out = fig_p1(wp)
    save_slide(fig_p1_out, "S12_weekly_promo")
    plt.close(fig_p1_out)

    print("7. Testing cutoff panel figures (V1, V3, L3, P2, Q1)...")
    panel = build_customer_cutoff_panel(tx)
    base_rf = rf_baseline(panel)
    panel["expected_rf"] = base_rf.panel["expected_rf"]

    print("V1: RF Heatmap...")
    fig_v1_out = fig_v1(base_rf.cells)
    save_slide(fig_v1_out, "S8_rf_heatmap")
    plt.close(fig_v1_out)

    print("L3: Cutoff return rates...")
    rates = cutoff_return_rates(panel)
    print(f"Cutoff rates median: {rates['return_rate'].median()*100:.2f}%")
    fig_l3_out = fig_l3(rates)
    save_slide(fig_l3_out, "S6_label_stability")
    plt.close(fig_l3_out)

    print("P2: Campaign reach by RF...")
    from propensity.features.promotion import campaign_features
    camp_desc = pd.read_csv("data/raw/Dunnhumby/campaign_desc.csv")
    cf = campaign_features(camp, camp_desc, coup, panel)
    from propensity.eda.propensity_rates import join_features
    panel = join_features(panel, cf)
    reach = campaign_reach_by_rf(panel)
    fig_p2_out = fig_p2(reach)
    save_slide(fig_p2_out, "S12_campaign_reach")
    plt.close(fig_p2_out)

    print("Q1: Demographic bias...")
    from propensity.features.demographic import group_demographic, demographic_features
    demo_grp = group_demographic(demo)
    panel = demographic_features(panel, demo_grp)
    demo_cov = demographic_coverage_profile(panel)
    fig_q1_out = fig_q1(demo_cov)
    save_slide(fig_q1_out, "S13_demographic_bias")
    plt.close(fig_q1_out)

    print("V3: Trend excess...")
    from propensity.features.temporal import temporal_features, trend_bin
    pdays_distinct = tx[tx["SALES_VALUE"] > 0][["household_key", "DAY"]].drop_duplicates()
    tf = temporal_features(pdays_distinct, panel["cutoff_day"].unique().tolist())
    tf["trend_bin"] = trend_bin(tf["recent_ipt_ratio_c"])
    panel = join_features(panel, tf[["household_key", "cutoff_day", "trend_bin"]])
    trend_tbl = add_verdicts(excess_over_rf(panel, "trend_bin"))
    print("Trend excess table:")
    print(trend_tbl[["trend_bin", "observed", "expected", "excess_pp", "verdict"]])
    fig_v3_out = fig_v3(trend_tbl)
    save_slide(fig_v3_out, "S10_trend_excess")
    plt.close(fig_v3_out)

    print("8. Testing V2: Dumbbell chart with feature_signal_summary...")
    from propensity.features.category import category_features, commodity_stats, define_staples, household_day_category
    from propensity.features.promotion import promo_features
    from propensity.features.basket import day_basket_table, basket_features
    from propensity.eda.propensity_rates import quantile_bin

    stats = commodity_stats(lines, cat_cfg)
    staples = define_staples(stats, cat_cfg)
    staple_col = "commodity" if "commodity" in staples.columns else "COMMODITY_DESC"
    staple_set = set(staples[staples["is_staple"]][staple_col].tolist())
    hdc = household_day_category(lines, staple_set)
    cat_feats = category_features(hdc, panel["cutoff_day"].unique().tolist())
    panel = join_features(panel, cat_feats)

    p_feats = promo_features(lines, panel["cutoff_day"].unique().tolist())
    panel = join_features(panel, p_feats)

    day_tbl = day_basket_table(baskets, lines=lines)
    bf = basket_features(day_tbl, panel["cutoff_day"].unique().tolist())
    panel = join_features(panel, bf)

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
    from propensity.features.temporal import lapse_bin
    panel["lapse_bin"] = lapse_bin(panel["lapse_ratio_c"])

    results = {
        "Xu hướng nhịp mua": trend_tbl,
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
    print(summary_df)
    fig_v2_out = fig_v2(summary_df)
    save_slide(fig_v2_out, "S9_raw_vs_rf")
    plt.close(fig_v2_out)

    print("ALL 13 SLIDE FIGURES GENERATED SUCCESSFULLY!")

if __name__ == "__main__":
    main()

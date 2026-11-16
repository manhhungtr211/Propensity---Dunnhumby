# Purchase propensity dataset: T30 / H7 / step14

Generated from all Dunnhumby source tables by:

```powershell
uv run python scripts/build_purchase_feature_dataset.py
```

## Grain and time policy

- Grain: one row per `(household_key, cutoff_day)`.
- Cutoffs: DAY 31 through DAY 703, spaced 14 days apart.
- Observation window: `[cutoff_day - 29, cutoff_day]`.
- Target window: `(cutoff_day, cutoff_day + 7]`.
- `target = 1` when the household has any valid purchase in the target window.
- Valid purchase: `QUANTITY > 0` and `SALES_VALUE >= 0`; zero-value purchases are retained.
- Households enter the panel from their first valid purchase onward.

## Files

### `household_cutoff_features.parquet`

Contains 114,685 rows, 2,500 households, 49 cutoffs, target metadata, exposure fields,
and leakage-safe features from all source tables:

- activity: product lines, baskets, active days/weeks;
- diversity: unique products and stores;
- value: sales, quantity, discount, average basket value;
- composition: discounted-line share and average lines per basket;
- recency: days since last purchase;
- product hierarchy: departments, commodities, subcommodities, manufacturers,
  and private-label share;
- demographics: original categories plus explicit numeric encodings;
- marketing: assigned/active campaigns, coupon availability, and redemption history;
- causal promotion context: transaction-line shares with display or mailer support;
- eligibility diagnostics: exposure days and full-window flag.

Campaign features only use campaigns whose `START_DAY <= cutoff_day`; redemption and
transaction features only use events on or before the cutoff. Causal exposure is joined
at `(PRODUCT_ID, STORE_ID, WEEK_NO)`. Because `causal_data.csv` is large, the builder
first semi-joins it to keys that actually occur in valid transactions.

### `feature_target_associations.parquet`

Contains point-biserial correlation between each numeric feature and binary target at
each cutoff. Only rows with `has_full_observation_window = true` enter the calculation.
`used_for_summary` additionally requires at least 300 non-null observations. Median and
P10-P90 correlations are repeated per feature for convenient plotting.

These correlations are descriptive, not causal. Modeling and validation must split by
cutoff time rather than randomly splitting repeated household rows.

### `feature_correlation_matrix.csv` and `feature_correlation_heatmap.png`

The full feature-feature-target Pearson matrix. Correlations are calculated independently
at each qualifying cutoff and then summarized by the median, so the heatmap is not a pooled
correlation dominated by cohort growth or time effects. Regenerate with:

```powershell
uv run python scripts/plot_feature_correlation_heatmap.py
```

# who: will this firm buy in the next 12 months?

Predicts whether a firm buys a vehicle in the next 12 months. In which of
those months is predicted by [when](../when/README.md).
Overview and results: [main README](../../../README.md).

## Model

An **XGBoost classifier** (`XGBClassifier`, binary) with one row per firm at a
cutoff date. The label is 1 if the firm bought in the 12 months from the
cutoff.

- **Train:** features at the cutoff two years back, labels from the year after it.
- **Validation:** features at the cutoff one year back. This year sets the
  threshold for the yes/no call and fits the calibration.
- **Calibration:** isotonic regression maps the raw scores to observed buying
  rates. The buyer weighting below inflates the raw scores, and calibration
  removes that.
- **Yes/no call:** the threshold that finds the most buyers while calling at
  most 20 % of non-buyers. It is chosen on the raw scores.

## Hyperparameters

| Parameter | Value | Note |
|---|---|---|
| `n_estimators` | 300 | fixed: the validation set is passed to `fit` but does not stop training early |
| `max_depth` | 6 | |
| `learning_rate` | 0.05 | |
| `subsample` | 0.8 | rows sampled per tree |
| `colsample_bytree` | 0.8 | features sampled per tree |
| `scale_pos_weight` | non-buyers / buyers | weights buyers up to balance the classes |
| `random_state` | 42 | |

## Features

All 33 features are computed from purchases strictly before the cutoff.
Missing values become 0.

| Group | Features |
|---|---|
| Volume | `total_purchases`, `purchases_last_1y`, `purchases_last_2y`, `purchases_last_5y`, `purchases_prev_3y` (24 to 60 months back) |
| Timing | `months_since_last`, `tenure_years` (since the first purchase), `active_years` (calendar years with a purchase) |
| Rate and trend | `purchase_frequency` (purchases per active year), `purchase_trend` (last 2 years / the 3 years before) |
| Gaps | `avg_interval_days`, `median_interval_days`. Gaps under 30 days count as one order delivered in parts and are dropped |
| Season | `month_pct_1` to `month_pct_12`: the share of past purchases in each calendar month |
| Brand | `distinct_brands`, `loyalty_score` (share of purchases going to the most bought brand), `brand_switches`, `top_brand_encoded` (most bought brand; a brand unseen in training becomes -1) |
| Fleet | `vehicle_age_avg`, `vehicle_age_min`, `vehicle_age_max` (cutoff year minus year built), `weight_avg`, `weight_max` |

## Files

- `features.py`: cutoff dates, the feature table and the brand encoding.
- `model.py`: `fit()` and `WhoModel.predict()`.
- `threshold.py`: the yes/no threshold and the metrics at it.

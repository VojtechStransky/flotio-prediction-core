# month_prediction

Predicts in which of the coming months a firm buys a vehicle. The output is each
month's share of the firm's purchase chance; the 12-month chance itself comes
from [year_prediction](../year_prediction/README.md). Overview and results:
[main README](../../../README.md).

## Model

An **XGBoost classifier** (`XGBClassifier`, binary) used as a discrete-time
hazard model: one row per firm × month-start × months-ahead `h`, label 1 if the
firm bought in month start + `h`. One model covers all horizons; `h` and the
target calendar month are inputs.

- **Training rows:** every month-start from 2016-01 up to `data_end`, horizons
  0 to 14. A label is used only when its month is before `data_end`; later
  months are unknown, not "did not buy".
- **Sampling:** all buyer rows and 10 % of non-buyer rows are kept. Predicted
  odds are multiplied by 0.10 to undo the sampling, so the output is a real
  monthly probability.
- **Firms modelled:** those with a purchase in the last 60 months.
- **Output:** probability per month for 15 months (12 shown + up to 3 months of
  lag in the source data). `month_shares()` divides each by their sum over the
  requested months.
- **Gate:** used only if `backtest()` shows it places the month at least as
  well as the hand-written rule below; otherwise the rule stays.

## Hyperparameters

| Parameter | Value | Note |
|---|---|---|
| `n_estimators` | 400 | fixed, no validation set and no early stopping |
| `max_depth` | 5 | |
| `learning_rate` | 0.05 | |
| `subsample` | 0.8 | rows sampled per tree |
| `colsample_bytree` | 0.8 | features sampled per tree |
| `min_child_weight` | 20 | |
| `tree_method` | `hist` | |
| `random_state` | 0 | also seeds the non-buyer sampling |

No class weights: the 10 % sampling balances the classes instead.

## Features

The 11 history features are computed from purchases strictly before the
month-start. Missing values stay missing (XGBoost handles them), e.g. the gaps
of a firm with purchases in only one month.

| Group | Features |
|---|---|
| Recency | `days_since` (since the last purchase) |
| Volume | `n_3m`, `n_6m`, `n_12m`, `n_24m`, `n_60m` (purchases in the last k months), `n_total`, `buymonths` (distinct months with a purchase) |
| Gaps | `med_gap_days`, `last_gap_days`: median and last gap between purchase months, in days |
| Due | `overdue` = `days_since` / `med_gap_days`, above 1 means past the usual gap |
| Target month | `h` (months ahead, 0 to 14), `tm` (calendar month of the target, 1 to 12) |

## Hand-written rule

The baseline the model has to beat (`rule_profile()`), computed per firm from
purchases before the month-start. The score of month start + `h` is

```
calendar share × due boost × (1 − 0.03·h)
```

- **Calendar share:** the share of the firm's past purchases that fell in that
  calendar month; 1/12 for a month it never bought in.
- **Due boost:** the next purchase is expected one average gap (mean days
  between purchases) after the last one. ×1.5 within 2 months of that date,
  ×0.7 more than 3 months past it, ×1 otherwise. Firms with a single purchase
  get no boost.
- **Decay:** 3 % less per month ahead.
- A firm with no history gets a flat 1/12 for every month.

## Files

- `history.py`: month indexing, the per-firm inputs and the month labels.
- `model.py`: `train()` and `HazardModel.predict()`.
- `shares.py`: `month_shares()`, each month's share of a firm's chance.
- `rule.py`: `rule_profile()`, the hand-written rule.
- `backtest.py`: `backtest()`, the gate: model vs rule vs random window.

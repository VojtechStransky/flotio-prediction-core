"""WHEN: month-by-month purchase model (discrete-time hazard).

The 12-month classifier in year_prediction answers *who* buys. This model answers
*which month*: for a firm at month start O it predicts p_h = P(firm buys a
vehicle in month O+h) for h = 0..H-1. Its share of each month (p_h / sum p)
is meant to replace a hand-written rule (month share of past purchases x
due-date boost x decay).

Design:
  * One stacked XGBoost: rows are firm x month-start x months-ahead, with
    months-ahead and the target calendar month as inputs.
  * Inputs are purchase-history only (recency, counts, gaps), computed from
    purchases strictly before the snapshot, so every past snapshot is exactly
    what was knowable then.
  * A label is used only when its month is before `data_end`.
  * 10% of non-buyer rows are kept; probabilities are corrected for the sampling.

`backtest()` retrains with an earlier cutoff and checks month placement on months
whose outcome is known, against the hand-written rule (rule_profile) and a random
window. The caller uses the shares only if they place the month at least as well
as the rule; otherwise it keeps the rule.

Files, by usage:
  history.py   month indexing, purchase history -> per-firm inputs and month labels
  model.py     the hazard model: train() and HazardModel.predict()
  shares.py    month_shares(): what callers use, each month's share of a firm's chance
  rule.py      rule_profile(): the hand-written rule, the baseline to beat
  backtest.py  backtest(): the gate, model vs rule vs random on known months
"""

from .backtest import backtest
from .history import (
    ACTIVE_MONTHS,
    FEATURES,
    HORIZON,
    features_at,
    labels_at,
    month_index,
    month_start,
    purchase_months,
)
from .model import FIRST_ORIGIN, INPUTS, NEG_RATE, HazardModel, train
from .rule import rule_profile
from .shares import month_shares

__all__ = [
    "ACTIVE_MONTHS",
    "FEATURES",
    "FIRST_ORIGIN",
    "HORIZON",
    "INPUTS",
    "NEG_RATE",
    "HazardModel",
    "backtest",
    "features_at",
    "labels_at",
    "month_index",
    "month_shares",
    "month_start",
    "purchase_months",
    "rule_profile",
    "train",
]

"""WHO: probability that a firm buys a vehicle in the next 12 months.

One row per firm at a cutoff date. Every input is computed from purchases
strictly before the cutoff; the label is whether the firm bought in the 12
months from it. The same function builds the training, validation and
prediction tables, so the three can only differ by the cutoff.

Flow (see fit):
  train    features at cutoff T-24m, labels from the year after
  validate features at cutoff T-12m: operating threshold and probability
           calibration (watched during training, but no early stopping)
  predict  features at cutoff T

Input table, one row per purchased vehicle:
  firm (int), purchase_date (date), brand (str), year_built (int), weight (float)

Files, by usage:
  features.py   cutoff dates, purchase history -> per-firm inputs and the 12-month label, brand encoding
  model.py      the classifier: fit() trains and calibrates it, WhoModel.predict() scores firms
  threshold.py  the yes/no call: metrics at a threshold, the threshold chosen by its cost
"""

from .features import FEATURES, add_months, build_features, prepare_xy
from .model import WhoModel, calibrate_model, fit, train_model
from .threshold import metrics_at_threshold, threshold_at_max_recall_under_fpr

__all__ = [
    "FEATURES",
    "WhoModel",
    "add_months",
    "build_features",
    "calibrate_model",
    "fit",
    "metrics_at_threshold",
    "prepare_xy",
    "threshold_at_max_recall_under_fpr",
    "train_model",
]

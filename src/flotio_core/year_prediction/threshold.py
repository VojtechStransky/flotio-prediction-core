"""The yes/no call: metrics at a threshold and the threshold chosen by its cost."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def metrics_at_threshold(y_true, y_proba, threshold: float) -> dict:
    y_pred = y_proba >= threshold
    tp = int((y_pred & (y_true == 1)).sum())
    fp = int((y_pred & (y_true == 0)).sum())
    fn = int((~y_pred & (y_true == 1)).sum())
    tn = int((~y_pred & (y_true == 0)).sum())
    return {
        "threshold": float(threshold),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def threshold_at_max_recall_under_fpr(y_true, y_proba, max_fpr: float = 0.20) -> tuple[float, dict]:
    """The threshold that finds the most buyers while calling at most `max_fpr` of non-buyers.

    A sales team would sooner call a few firms that do not buy than miss ones
    that do. So the share of wasted calls is capped, and within that budget the
    threshold with the highest recall is taken.
    """
    fpr, _, thresholds = roc_curve(y_true, y_proba)
    allowed = np.where(fpr <= max_fpr)[0]
    threshold = float(thresholds[allowed[-1]]) if len(allowed) else float(thresholds[0])
    return threshold, metrics_at_threshold(y_true, y_proba, threshold)

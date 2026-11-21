"""backtest(): the gate. Is the month model at least as good as the hand-written rule?"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from .history import labels_at, month_index, month_start, purchase_months
from .model import train
from .rule import rule_profile


def backtest(purchases: pl.DataFrame, cutoff: date, data_end: date) -> dict:
    """Retrain with labels before `cutoff`, then for 12 month-starts from `cutoff`
    (outcomes known before `data_end`) measure, among firms that bought in months
    1..6, how often the model's 3-month window (centred on months 2..5) contains a
    purchase, against the hand-written rule and a window chosen at random."""
    hm = train(purchases, cutoff)
    pm = purchase_months(purchases)
    end_mi, c_mi = month_index(data_end), month_index(cutoff)
    model_hits, rule_hits, random_hits, n = 0.0, 0.0, 0.0, 0
    for omi in range(c_mi, c_mi + 12):
        if omi + 7 >= end_mi:
            break
        firms, ph = hm.predict(pm, month_start(omi))
        if len(firms) == 0:
            continue
        y = np.nan_to_num(labels_at(pm, firms, omi, end_mi))
        buy = y[:, 1:7].sum(1) > 0
        hit = np.stack([y[:, c - 1 : c + 2].max(1) for c in range(2, 6)], 1)
        win = np.stack([1 - np.prod(1 - ph[:, c - 1 : c + 2], 1) for c in range(2, 6)], 1)
        rp = rule_profile(pm, firms, omi)
        rwin = np.stack([rp[:, c - 1 : c + 2].sum(1) for c in range(2, 6)], 1)
        rows = np.arange(len(hit))
        model_hits += hit[rows, win.argmax(1)][buy].sum()
        rule_hits += hit[rows, rwin.argmax(1)][buy].sum()
        random_hits += hit[buy].mean(1).sum()  # exact expectation over the four windows
        n += int(buy.sum())
    if n == 0:
        return {"buyers": 0, "passed": False}
    model_pct, rule_pct, random_pct = (100 * model_hits / n, 100 * rule_hits / n, 100 * random_hits / n)
    return {
        "cutoff": str(cutoff),
        "buyers": n,
        "model_pct": round(float(model_pct), 2),
        "hand_rule_pct": round(float(rule_pct), 2),
        "random_pct": round(float(random_pct), 2),
        "gain_vs_rule_pts": round(float(model_pct - rule_pct), 2),
        "gain_vs_random_pts": round(float(model_pct - random_pct), 2),
        "passed": bool(model_pct >= rule_pct),
    }

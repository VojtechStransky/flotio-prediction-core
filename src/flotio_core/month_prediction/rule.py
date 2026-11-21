"""The hand-written monthly rule: the baseline the month model has to beat."""

from __future__ import annotations

import numpy as np
import polars as pl

from .history import HORIZON, month_start


def rule_profile(pm: pl.DataFrame, firms: pl.Series, origin_mi: int, horizon: int = HORIZON) -> np.ndarray:
    """The hand-written monthly rule, from purchases
    before the snapshot: share of the firm's past purchases in that calendar month
    (1/12 for months it never bought in) x 1.5 near the usual gap's due date
    (x 0.7 well past it) x a 3%-per-month decay."""
    p = pm.filter(pl.col("mi") < origin_mi)
    o = month_start(origin_mi)
    per = (
        p.sort("firm", "d")
        .group_by("firm")
        .agg(
            (pl.col("mi") % 12 + 1).alias("cal"),
            pl.col("d").diff().dt.total_days().mean().alias("avg_gap"),
            (pl.lit(o) - pl.col("d").max()).dt.total_days().alias("since"),
        )
    )
    rows = {r[0]: r for r in per.iter_rows()}
    out = np.full((len(firms), horizon), 1 / 12)
    for i, f in enumerate(firms.to_list()):
        r = rows.get(f)
        if r is None:
            continue
        cal = np.bincount(np.asarray(r[1]) - 1, minlength=12) / len(r[1])
        due = max(0.0, (r[2] or 0) / 30 - r[3] / 30.44) if r[2] else None
        for h in range(horizon):
            share = cal[(origin_mi + h) % 12] or 1 / 12
            boost = 1.0
            if due is not None:
                boost = 1.5 if abs(h - due) < 2 else (0.7 if h > due + 3 else 1.0)
            out[i, h] = share * boost * (1 - 0.03 * h)
    return out

"""Purchase history -> per-firm inputs and month labels, shared by the model, the rule and the gate."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

FEATURES = [
    "days_since",
    "n_3m",
    "n_6m",
    "n_12m",
    "n_24m",
    "n_60m",
    "n_total",
    "buymonths",
    "med_gap_days",
    "last_gap_days",
    "overdue",
]
HORIZON = 15  # 12 shown months + up to 3 months of lag in the source data
ACTIVE_MONTHS = 60  # firms with no purchase in 5 years are not modelled


def month_index(d: date) -> int:
    return d.year * 12 + d.month - 1


def month_start(mi: int) -> date:
    return date(mi // 12, mi % 12 + 1, 1)


def purchase_months(purchases: pl.DataFrame, firm_col: str = "firm", date_col: str = "purchase_date") -> pl.DataFrame:
    """One row per purchase with firm, date and month index."""
    return (
        purchases.select(pl.col(firm_col).cast(pl.Int64).alias("firm"), pl.col(date_col).alias("d"))
        .drop_nulls()
        .with_columns(
            (pl.col("d").dt.year().cast(pl.Int32) * 12 + pl.col("d").dt.month().cast(pl.Int32) - 1).alias("mi")
        )
    )


def features_at(pm: pl.DataFrame, origin_mi: int) -> pl.DataFrame:
    """Per-firm inputs at the start of month `origin_mi`, from purchases before it."""
    o = month_start(origin_mi)
    p = pm.filter(pl.col("mi") < origin_mi)
    hist = p.group_by("firm").agg(
        (pl.lit(o) - pl.col("d").max()).dt.total_days().alias("days_since"),
        *[(pl.col("mi") >= origin_mi - k).sum().alias(f"n_{k}m") for k in (3, 6, 12, 24, 60)],
        pl.len().alias("n_total"),
        pl.col("mi").n_unique().alias("buymonths"),
    )
    gaps = (
        p.select("firm", "mi")
        .unique()
        .sort("firm", "mi")
        .with_columns(((pl.col("mi") - pl.col("mi").shift(1).over("firm")) * 30.44).alias("g"))
        .group_by("firm")
        .agg(pl.col("g").median().alias("med_gap_days"), pl.col("g").drop_nulls().last().alias("last_gap_days"))
    )
    df = hist.join(gaps, on="firm", how="left").filter(pl.col("days_since") <= ACTIVE_MONTHS * 30.44)
    return df.with_columns((pl.col("days_since") / pl.col("med_gap_days")).alias("overdue"))


def labels_at(pm: pl.DataFrame, firms: pl.Series, origin_mi: int, data_end_mi: int) -> np.ndarray:
    """(n_firms, HORIZON) float array: 1/0 bought in month origin+h, NaN if not yet known."""
    fut = (
        pm.filter(pl.col("mi").is_between(origin_mi, origin_mi + HORIZON - 1))
        .select("firm", (pl.col("mi") - origin_mi).alias("h"))
        .unique()
    )
    idx = {f: i for i, f in enumerate(firms.to_list())}
    y = np.zeros((len(idx), HORIZON), dtype=np.float32)
    for f, h in fut.iter_rows():
        i = idx.get(f)
        if i is not None:
            y[i, h] = 1.0
    known = np.array([origin_mi + h < data_end_mi for h in range(HORIZON)])
    y[:, ~known] = np.nan
    return y

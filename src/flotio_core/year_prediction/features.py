"""Purchase history -> one row per firm at a cutoff: the inputs, the 12-month label and the brand encoding."""

from __future__ import annotations

from datetime import date

import polars as pl
from sklearn.preprocessing import LabelEncoder

FEATURES = [
    "total_purchases",
    "months_since_last",
    "tenure_years",
    "active_years",
    "purchase_frequency",
    "purchase_trend",
    "purchases_last_1y",
    "purchases_last_2y",
    "purchases_last_5y",
    "purchases_prev_3y",
    "vehicle_age_avg",
    "vehicle_age_min",
    "vehicle_age_max",
    "distinct_brands",
    "loyalty_score",
    "brand_switches",
    "weight_avg",
    "weight_max",
    "avg_interval_days",
    "median_interval_days",
    *[f"month_pct_{m}" for m in range(1, 13)],
]


def add_months(d: date, months: int) -> date:
    """First day of the month `months` after (or before) the month of `d`."""
    mi = d.year * 12 + d.month - 1 + months
    return date(mi // 12, mi % 12 + 1, 1)


def build_features(purchases: pl.DataFrame, cutoff: date) -> pl.DataFrame:
    """One row per firm with a purchase before `cutoff`: inputs and the 12-month label.

    `target` is 1 when the firm bought in [cutoff, cutoff + 12 months). It is 0
    for every firm when the data ends before the cutoff, which is the case when
    predicting.
    """
    cutoff = add_months(cutoff, 0)
    historical = purchases.filter(pl.col("purchase_date") < cutoff)
    future = purchases.filter((pl.col("purchase_date") >= cutoff) & (pl.col("purchase_date") < add_months(cutoff, 12)))
    target_buyers = future["firm"].unique()

    def since(months: int) -> pl.Expr:
        return pl.col("purchase_date") >= add_months(cutoff, -months)

    age = cutoff.year - pl.col("year_built")
    firms = historical.group_by("firm").agg(
        pl.len().alias("total_purchases"),
        pl.col("purchase_date").max().alias("last_purchase"),
        pl.col("purchase_date").min().alias("first_purchase"),
        pl.col("purchase_date").dt.year().n_unique().alias("active_years"),
        since(12).sum().alias("purchases_last_1y"),
        since(24).sum().alias("purchases_last_2y"),
        since(60).sum().alias("purchases_last_5y"),
        (since(60) & ~since(24)).sum().alias("purchases_prev_3y"),
        age.mean().alias("vehicle_age_avg"),
        age.min().alias("vehicle_age_min"),
        age.max().alias("vehicle_age_max"),
        pl.col("brand").n_unique().alias("distinct_brands"),
        pl.col("brand").mode().sort().first().alias("top_brand"),
        pl.col("weight").mean().alias("weight_avg"),
        pl.col("weight").max().alias("weight_max"),
        # share of past purchases in each calendar month
        *[(pl.col("purchase_date").dt.month() == m).mean().alias(f"month_pct_{m}") for m in range(1, 13)],
    )

    firms = firms.with_columns(
        ((pl.lit(cutoff) - pl.col("last_purchase")).dt.total_days() / 30.44).alias("months_since_last"),
        ((pl.lit(cutoff) - pl.col("first_purchase")).dt.total_days() / 365.25).alias("tenure_years"),
        (pl.col("total_purchases") / pl.col("active_years").clip(1, None)).alias("purchase_frequency"),
        (pl.col("purchases_last_2y") / pl.col("purchases_prev_3y").clip(1, None)).alias("purchase_trend"),
        pl.col("firm").is_in(target_buyers.implode()).cast(pl.Int32).alias("target"),
    )

    # Loyalty: share of the firm's purchases that went to its most bought brand.
    loyalty = (
        historical.group_by("firm", "brand")
        .len()
        .group_by("firm")
        .agg((pl.col("len").max() / pl.col("len").sum()).alias("loyalty_score"))
    )

    # Brand switches: purchases whose brand differs from the previous purchase.
    ordered = historical.sort("firm", "purchase_date")
    switches = (
        ordered.with_columns(pl.col("brand").shift(1).over("firm").alias("prev_brand"))
        .filter(pl.col("prev_brand").is_not_null() & (pl.col("prev_brand") != pl.col("brand")))
        .group_by("firm")
        .len()
        .rename({"len": "brand_switches"})
    )

    # Gaps between purchases; gaps under 30 days are one order delivered in parts.
    intervals = (
        ordered.with_columns(
            (pl.col("purchase_date") - pl.col("purchase_date").shift(1).over("firm")).dt.total_days().alias("gap")
        )
        .filter(pl.col("gap") >= 30)
        .group_by("firm")
        .agg(
            pl.col("gap").mean().alias("avg_interval_days"),
            pl.col("gap").median().alias("median_interval_days"),
        )
    )

    return (
        firms.join(loyalty, on="firm", how="left")
        .join(switches, on="firm", how="left")
        .join(intervals, on="firm", how="left")
        .with_columns(pl.col("brand_switches").fill_null(0))
        .sort("firm")
    )


def prepare_xy(features: pl.DataFrame, encoder: LabelEncoder | None = None):
    """Turn a feature table into (X, y, firms, feature_names, encoder).

    The brand encoder is fitted on the training table only (encoder=None) and
    reused for every later table. Fitting it again would give the same brand a
    different number than the model was trained on. A brand the encoder has
    not seen becomes -1, so a new brand cannot stop a prediction run.
    """
    brands = features["top_brand"].fill_null("UNKNOWN").to_list()

    if encoder is None:
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(brands).tolist()
    else:
        known = {b: i for i, b in enumerate(encoder.classes_)}
        encoded = [known.get(b, -1) for b in brands]

    feature_names = [*FEATURES, "top_brand_encoded"]
    table = features.with_columns(pl.Series("top_brand_encoded", encoded)).with_columns(
        [pl.col(c).fill_null(0).cast(pl.Float64) for c in feature_names]
    )

    X = table.select(feature_names).to_numpy()
    y = table["target"].to_numpy()
    return X, y, table["firm"], feature_names, encoder

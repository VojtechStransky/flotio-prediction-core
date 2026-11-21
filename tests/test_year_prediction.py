"""WHO model: nothing from after the cutoff may reach the inputs, and the
brand encoding must stay the one the model was trained with."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from flotio_core import year_prediction


def _purchases(rows):
    return pl.DataFrame(
        [{"firm": f, "purchase_date": d, "brand": b, "year_built": d.year - 1, "weight": w} for f, d, b, w in rows],
        schema={
            "firm": pl.Int64,
            "purchase_date": pl.Date,
            "brand": pl.Utf8,
            "year_built": pl.Int32,
            "weight": pl.Float64,
        },
    )


@pytest.fixture
def purchases():
    rows = [(1, date(y, 6, 1), "BRAND_A", 1000.0) for y in (2021, 2022, 2022, 2023, 2023)]
    # Firm 1 after the cutoff: a brand never seen before and a weight used as a canary.
    rows.append((1, date(2025, 3, 1), "BRAND_NEW", 9999.0))
    rows += [(2, date(y, 9, 1), "BRAND_B", 2000.0) for y in (2021, 2022, 2023)]
    return _purchases(rows)


CUTOFF = date(2025, 1, 1)


def test_a_purchase_after_the_cutoff_does_not_reach_the_inputs(purchases):
    firm1 = year_prediction.build_features(purchases, CUTOFF).filter(pl.col("firm") == 1).row(0, named=True)
    assert firm1["total_purchases"] == 5
    assert firm1["weight_max"] == pytest.approx(1000.0)  # not the 9999 canary
    assert firm1["weight_avg"] == pytest.approx(1000.0)
    assert firm1["top_brand"] == "BRAND_A"
    assert firm1["distinct_brands"] == 1


def test_the_label_is_the_twelve_months_from_the_cutoff(purchases):
    features = year_prediction.build_features(purchases, CUTOFF)
    assert dict(zip(features["firm"], features["target"], strict=True)) == {1: 1, 2: 0}
    # A year earlier firm 1 bought nothing in 2024, and its 2025 purchase is out of the window.
    earlier = year_prediction.build_features(purchases, date(2024, 1, 1))
    assert dict(zip(earlier["firm"], earlier["target"], strict=True)) == {1: 0, 2: 0}


def test_a_purchase_on_the_cutoff_day_is_the_future(purchases):
    on_the_day = pl.concat([purchases, _purchases([(2, CUTOFF, "BRAND_B", 2000.0)])])
    firm2 = year_prediction.build_features(on_the_day, CUTOFF).filter(pl.col("firm") == 2).row(0, named=True)
    assert firm2["total_purchases"] == 3 and firm2["target"] == 1


def test_recency_and_gaps_are_measured_to_the_cutoff(purchases):
    firm2 = year_prediction.build_features(purchases, CUTOFF).filter(pl.col("firm") == 2).row(0, named=True)
    assert firm2["months_since_last"] == pytest.approx((CUTOFF - date(2023, 9, 1)).days / 30.44)
    assert firm2["purchases_last_1y"] == 0 and firm2["purchases_last_2y"] == 1
    assert firm2["median_interval_days"] == 365
    assert firm2["month_pct_9"] == 1.0 and firm2["month_pct_6"] == 0.0


def test_the_brand_encoder_is_reused_and_an_unseen_brand_is_minus_one(purchases):
    train = year_prediction.build_features(purchases, CUTOFF)
    X_train, _, _, names, encoder = year_prediction.prepare_xy(train)
    column = names.index("top_brand_encoded")

    # After the cutoff moves, firm 1 is still BRAND_A (5 of 6) and keeps its code.
    later = year_prediction.build_features(purchases, date(2026, 1, 1))
    X_later, _, _, _, same_encoder = year_prediction.prepare_xy(later, encoder)
    assert same_encoder is encoder
    assert X_later[:, column].tolist() == X_train[:, column].tolist()

    newcomer = pl.concat([purchases, _purchases([(3, date(2025, 5, 1), "BRAND_NEW", 500.0)])])
    X_new, _, firms, _, _ = year_prediction.prepare_xy(
        year_prediction.build_features(newcomer, date(2026, 1, 1)), encoder
    )
    assert X_new[firms.to_list().index(3), column] == -1


def test_the_threshold_respects_the_false_positive_budget():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.15).astype(int)
    score = np.clip(0.25 * y + rng.normal(0.3, 0.15, 4000), 0, 1)
    threshold, point = year_prediction.threshold_at_max_recall_under_fpr(y, score, max_fpr=0.20)
    assert point["false_positive_rate"] <= 0.20
    # A lower threshold would find more buyers but break the budget.
    looser = year_prediction.metrics_at_threshold(y, score, threshold - 0.02)
    assert looser["recall"] >= point["recall"] and looser["false_positive_rate"] > 0.20


def test_metrics_count_the_four_outcomes():
    y = np.array([1, 1, 0, 0, 0])
    point = year_prediction.metrics_at_threshold(y, np.array([0.9, 0.2, 0.8, 0.1, 0.1]), 0.5)
    assert (point["tp"], point["fp"], point["fn"], point["tn"]) == (1, 1, 1, 2)
    assert point["precision"] == 0.5 and point["recall"] == 0.5

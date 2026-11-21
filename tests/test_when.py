"""WHEN model: point-in-time inputs, labels, and the month shares."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from flotio_core import when


def _purchases(rows):
    return pl.DataFrame(
        {"firm": [r[0] for r in rows], "purchase_date": [r[1] for r in rows]},
        schema={"firm": pl.Int64, "purchase_date": pl.Date},
    )


def test_features_only_use_purchases_before_the_snapshot():
    pm = when.purchase_months(_purchases([(1, date(2024, 1, 15)), (1, date(2024, 6, 3)), (1, date(2024, 7, 20))]))
    f = when.features_at(pm, when.month_index(date(2024, 7, 1))).row(0, named=True)
    assert f["n_total"] == 2  # the July purchase is not known on 1 July
    assert f["days_since"] == (date(2024, 7, 1) - date(2024, 6, 3)).days
    assert f["buymonths"] == 2 and f["n_3m"] == 1


def test_inactive_firms_are_not_modelled():
    pm = when.purchase_months(_purchases([(1, date(2015, 1, 1)), (2, date(2023, 1, 1))]))
    assert when.features_at(pm, when.month_index(date(2024, 1, 1)))["firm"].to_list() == [2]


def test_labels_are_unknown_from_data_end():
    pm = when.purchase_months(_purchases([(1, date(2024, 3, 9)), (1, date(2024, 5, 1))]))
    o, end = when.month_index(date(2024, 1, 1)), when.month_index(date(2024, 5, 1))
    y = when.labels_at(pm, pl.Series([1]), o, end)
    assert y[0, 2] == 1.0 and y[0, 0] == 0.0
    assert np.isnan(y[0, 4])  # May is at data_end: not yet known


@pytest.fixture(scope="module")
def seasonal_purchases():
    # Firms that buy every spring (April) and firms that buy every autumn (October).
    rows = []
    for firm in range(60):
        month = 4 if firm % 2 else 10
        for year in range(2014, 2025):
            rows.append((firm, date(year, month, 10)))
    return _purchases(rows)


def test_month_model_learns_a_seasonal_shape(seasonal_purchases):
    model = when.train(seasonal_purchases, date(2025, 1, 1), n_estimators=60)
    months = [date(2025, m, 1) for m in range(1, 13)]
    shares = when.month_shares(model, seasonal_purchases, months)
    assert len(shares) == 60
    for firm, w in shares.items():
        assert w.sum() == pytest.approx(1.0)
        assert months[int(np.argmax(w))].month == (4 if firm % 2 else 10)


def test_shares_outside_the_horizon_raise(seasonal_purchases):
    model = when.train(seasonal_purchases, date(2025, 1, 1), n_estimators=10)
    with pytest.raises(ValueError, match="horizon"):
        when.month_shares(model, seasonal_purchases, [date(2026, 6, 1)])


def test_shares_score_at_the_given_origin_not_the_training_end(seasonal_purchases, monkeypatch):
    hm = when.train(seasonal_purchases, date(2024, 1, 1), n_estimators=10)
    months = [date(2025, 6, 1)]  # beyond the horizon counted from the training end

    with pytest.raises(ValueError, match="horizon"):
        when.month_shares(hm, seasonal_purchases, months)

    origins = []
    real_predict = when.HazardModel.predict
    monkeypatch.setattr(
        when.HazardModel,
        "predict",
        lambda self, pm, origin: origins.append(origin) or real_predict(self, pm, origin),
    )
    assert when.month_shares(hm, seasonal_purchases, months, origin=date(2025, 1, 1))
    assert origins == [date(2025, 1, 1)]


def test_backtest_gates_the_model_on_the_hand_written_rule(seasonal_purchases):
    result = when.backtest(seasonal_purchases, date(2023, 1, 1), date(2025, 1, 1))
    assert result["buyers"] > 0
    assert {"model_pct", "hand_rule_pct", "random_pct", "passed"} <= set(result)
    assert 0 < result["random_pct"] < 100
    assert result["passed"] == (result["model_pct"] >= result["hand_rule_pct"])

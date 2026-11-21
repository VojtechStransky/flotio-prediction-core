"""Evaluation: the metrics must equal what can be counted by hand."""

import numpy as np
import pytest

from flotio_core import evaluate


def test_threshold_metrics_match_a_hand_count():
    # 4 buyers and 6 non-buyers; threshold 0.5 calls 3 buyers and 2 non-buyers
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    raw = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.55, 0.3, 0.2, 0.1, 0.05])
    t = evaluate.classifier_report(y, raw, raw, 0.5, bootstrap=0)["threshold"]
    assert (t["tp"], t["fp"], t["fn"], t["tn"]) == (3, 2, 1, 4)
    assert t["accuracy"] == pytest.approx(0.7)
    assert t["precision"] == pytest.approx(3 / 5)
    assert t["recall"] == pytest.approx(3 / 4)
    assert t["specificity"] == pytest.approx(4 / 6)
    assert t["accuracy_calling_nobody"] == pytest.approx(0.6)


def test_a_perfect_ranking_scores_one_and_a_reversed_one_zero():
    y = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    good = evaluate.classifier_report(y, score, score, 0.5, bootstrap=0)
    bad = evaluate.classifier_report(y, -score, 1 - score, 0.5, bootstrap=0)
    assert good["roc_auc"] == 1.0 and good["pr_auc"] == 1.0
    assert bad["roc_auc"] == 0.0
    assert good["top_20pct"]["buyers_found"] == pytest.approx(0.5)  # 1 of 5 firms called, 1 of 2 buyers found
    assert good["top_20pct"]["lift"] == pytest.approx(1 / 0.4)


def test_predicting_the_base_rate_has_no_brier_skill():
    y = np.array([1, 0, 0, 0] * 25)
    base = np.full(len(y), 0.25)
    report = evaluate.classifier_report(y, np.arange(len(y)) % 7, base, 3, bootstrap=0)
    assert report["brier"] == pytest.approx(0.25 * 0.75)
    assert report["brier_skill"] == pytest.approx(0.0)
    assert report["calibration_error"] == pytest.approx(0.0, abs=0.06)


def test_month_placement_counts_hits_and_distance():
    # firm 0 bought in month 2, firm 1 in months 0 and 5
    y = np.array([[0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 1]], dtype=float)
    score = np.array(
        [
            [0.1, 0.2, 0.9, 0.3, 0.0, 0.0],  # first choice month 2: a hit
            [0.0, 0.0, 0.1, 0.9, 0.2, 0.3],
        ]
    )  # first choice month 3: 2 months from month 5
    p = evaluate._placement(y, score)
    assert p["top1"] == pytest.approx(0.5)
    assert p["top3"] == pytest.approx(1.0)  # firm 1's top three are months 3, 5, 4
    assert p["months_off_mean"] == pytest.approx(1.0)
    assert p["within_one_month"] == pytest.approx(0.5)

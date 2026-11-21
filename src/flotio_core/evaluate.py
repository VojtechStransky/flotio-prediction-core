"""How good are the two models? Standard metrics and the points for the usual curves.

classifier_report  the 12-month classifier on a table it was not fitted on:
                   ranking (buyer-above-non-buyer rate, average precision, lift), the yes/no call (accuracy,
                   precision, recall) and the probabilities (Brier score,
                   calibration).
walk_forward       the same report for several test years, each with a model
                   fitted only on the years before it.
month_report       the month model on months whose outcome is known: is the
                   monthly probability any good, and does it place the month
                   better than the hand-written rule or a random month.

Every function returns plain dicts and lists, so a report can be saved as JSON
and drawn later (tools/plot_report.py).
"""

from __future__ import annotations

from datetime import date
from math import comb

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from . import month_prediction, year_prediction


def _thin(*arrays: np.ndarray, points: int = 80) -> list[list[float]]:
    """Keep about `points` evenly spaced points of a curve, always with both ends."""
    n = len(arrays[0])
    keep = np.unique(np.linspace(0, n - 1, min(points, n)).round().astype(int))
    return [[round(float(v), 4) for v in a[keep]] for a in arrays]


def _calibration(y: np.ndarray, p: np.ndarray, groups: int = 10) -> list[dict]:
    """Rows sorted by predicted probability and cut into equal groups."""
    return [
        {"predicted": round(float(p[g].mean()), 5), "observed": round(float(y[g].mean()), 5), "rows": int(len(g))}
        for g in np.array_split(np.argsort(p, kind="stable"), groups)
        if len(g)
    ]


def _probability_scores(y: np.ndarray, p: np.ndarray) -> dict:
    """Brier score and log loss, each next to the score of always predicting the base rate."""
    base = float(y.mean())
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    brier, brier_base = float(np.mean((p - y) ** 2)), base * (1 - base)
    log_loss = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))
    log_loss_base = float(-(base * np.log(base) + (1 - base) * np.log(1 - base))) if 0 < base < 1 else 0.0
    rows = _calibration(y, p)
    error = sum(r["rows"] * abs(r["predicted"] - r["observed"]) for r in rows) / len(y)
    return {
        "brier": brier,
        "brier_base_rate": brier_base,
        "brier_skill": 1 - brier / brier_base if brier_base else 0.0,
        "log_loss": log_loss,
        "log_loss_base_rate": log_loss_base,
        "calibration_error": float(error),
        "mean_predicted": float(p.mean()),
        "mean_observed": base,
    }


def classifier_report(
    y, raw, calibrated, threshold: float, *, curves: bool = True, bootstrap: int = 500, seed: int = 0
) -> dict:
    """Metrics of the 12-month classifier on one held-out table.

    y           1 if the firm bought in the 12 months after the cutoff
    raw         model score; ranks the firms and carries the yes/no threshold
    calibrated  the probability shown to the user
    threshold   the operating threshold chosen a year earlier, on raw scores
    """
    y, raw, calibrated = np.asarray(y).astype(int), np.asarray(raw, float), np.asarray(calibrated, float)
    n, buyers = len(y), int(y.sum())
    base = buyers / n
    fpr, tpr, _ = roc_curve(y, raw)
    precision, recall, _ = precision_recall_curve(y, raw)

    point = year_prediction.metrics_at_threshold(y, raw, threshold)
    tp, fp, fn, tn = point["tp"], point["fp"], point["fn"], point["tn"]

    # Firms in the order a sales team would call them.
    order = np.argsort(-raw, kind="stable")
    found = np.cumsum(y[order]) / max(buyers, 1)
    called = np.arange(1, n + 1) / n

    def top(share: float) -> dict:
        k = max(int(round(share * n)), 1)
        hit_rate = float(y[order[:k]].mean())
        return {"buyers_found": float(found[k - 1]), "hit_rate": hit_rate, "lift": hit_rate / base if base else 0.0}

    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(bootstrap):
        pick = rng.integers(0, n, n)
        if 0 < y[pick].sum() < n:
            aucs.append(roc_auc_score(y[pick], raw[pick]))

    report = {
        "firms": n,
        "buyers": buyers,
        "base_rate": base,
        "roc_auc": float(roc_auc_score(y, raw)),
        "roc_auc_ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))] if aucs else None,
        "pr_auc": float(average_precision_score(y, raw)),
        "ks": float(np.max(tpr - fpr)),
        "threshold": {
            "called": tp + fp,
            "called_share": (tp + fp) / n,
            "accuracy": (tp + tn) / n,
            "accuracy_calling_nobody": 1 - base,
            "precision": point["precision"],
            "recall": point["recall"],
            "specificity": tn / max(tn + fp, 1),
            "false_positive_rate": point["false_positive_rate"],
            "lift": point["precision"] / base if base else 0.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "top_10pct": top(0.10),
        "top_20pct": top(0.20),
        **_probability_scores(y, calibrated),
    }
    if curves:
        report["curves"] = {
            "roc": dict(zip(("false_positive_rate", "recall"), _thin(fpr, tpr), strict=True)),
            "precision_recall": dict(zip(("recall", "precision"), _thin(recall[::-1], precision[::-1]), strict=True)),
            "gains": dict(
                zip(
                    ("called_share", "buyers_found"),
                    _thin(np.r_[0.0, called], np.r_[0.0, found], points=50),
                    strict=True,
                )
            ),
            "calibration": _calibration(y, calibrated),
        }
    return report


def evaluate_who(purchases: pl.DataFrame, test_cutoff: date, *, curves: bool = True) -> dict:
    """Fit on the two years before `test_cutoff` and report on the year after it.

    train table   cutoff test - 24 months (labels: the year after it)
    validation    cutoff test - 12 months: threshold and calibration
    test          cutoff test, used for nothing but this report
    """
    val_cutoff = year_prediction.add_months(test_cutoff, -12)
    model = year_prediction.fit(purchases, val_cutoff)
    X, y, _, names, _ = year_prediction.prepare_xy(
        year_prediction.build_features(purchases, test_cutoff), model.encoder
    )
    raw = model.model.predict_proba(X)[:, 1]
    report = classifier_report(y, raw, model.calibrator.predict(raw), model.threshold, curves=curves)
    importance = sorted(zip(names, model.model.feature_importances_.tolist(), strict=True), key=lambda t: -t[1])
    return {
        "train_cutoff": str(year_prediction.add_months(val_cutoff, -12)),
        "validation_cutoff": str(val_cutoff),
        "test_cutoff": str(year_prediction.add_months(test_cutoff, 0)),
        "validation_roc_auc": model.metrics["val_auc"],
        **report,
        "top_features": [{"feature": f, "importance": round(v, 4)} for f, v in importance[:8]],
    }


def walk_forward(purchases: pl.DataFrame, test_cutoffs: list[date]) -> list[dict]:
    """evaluate_who for each test year, without the curves."""
    return [evaluate_who(purchases, cutoff, curves=False) for cutoff in test_cutoffs]


def _placement(y: np.ndarray, score: np.ndarray) -> dict:
    """How well `score` (n firms x months) places the purchase month, for firms that bought."""
    rows = np.arange(len(y))
    best = score.argmax(1)
    top3 = np.argsort(-score, axis=1, kind="stable")[:, :3]
    months = np.arange(y.shape[1])
    distance = np.where(y > 0, np.abs(months[None, :] - best[:, None]), np.inf).min(1)
    return {
        "top1": float(y[rows, best].mean()),
        "top3": float(y[rows[:, None], top3].max(1).mean()),
        "months_off_mean": float(distance.mean()),
        "within_one_month": float((distance <= 1).mean()),
    }


def month_report(purchases: pl.DataFrame, cutoff: date, data_end: date, *, months: int = 12, seed: int = 0) -> dict:
    """The month model, trained with labels before `cutoff`, judged on later month-starts.

    For each of up to 12 month-starts from `cutoff` whose next `months` months are
    known, every modelled firm gets a probability for each of those months.

    monthly_probability  all firm x month rows: can the probability tell the months
                         with a purchase from those without? This mixes *who* with
                         *when*: a firm that buys often is likely in every month.
    placement            only firms that bought in the window: is the month the
                         model ranks first (or its three highest) a month with a
                         purchase? Compared with the hand-written rule and a
                         random month.
    """
    hm = month_prediction.train(purchases, cutoff)
    pm = month_prediction.purchase_months(purchases)
    end_mi, c_mi = month_prediction.month_index(data_end), month_prediction.month_index(cutoff)
    rng = np.random.default_rng(seed)

    ys, ps, rules, origins = [], [], [], 0
    for omi in range(c_mi, c_mi + 12):
        if omi + months > end_mi:
            break
        firms, ph = hm.predict(pm, month_prediction.month_start(omi))
        if len(firms) == 0:
            continue
        ys.append(month_prediction.labels_at(pm, firms, omi, end_mi)[:, :months])
        ps.append(ph[:, :months])
        rules.append(month_prediction.rule_profile(pm, firms, omi)[:, :months])
        origins += 1
    if not ys:
        return {"origins": 0}
    y, p, rule = np.concatenate(ys), np.concatenate(ps), np.concatenate(rules)

    flat_y, flat_p = y.ravel().astype(int), p.ravel()
    monthly = {
        "rows": int(len(flat_y)),
        "purchases": int(flat_y.sum()),
        "base_rate": float(flat_y.mean()),
        "roc_auc": float(roc_auc_score(flat_y, flat_p)),
        "pr_auc": float(average_precision_score(flat_y, flat_p)),
        **_probability_scores(flat_y, flat_p),
        "calibration": _calibration(flat_y, flat_p),
    }

    buy = y.sum(1) > 0
    yb, k = y[buy], y[buy].sum(1).astype(int)
    random_scores = rng.random(yb.shape)
    placement = {
        "firms_that_bought": int(buy.sum()),
        "purchase_months_per_firm": float(k.mean()),
        "model": _placement(yb, p[buy]),
        "hand_rule": _placement(yb, rule[buy]),
        "random": {
            **_placement(yb, random_scores),
            # exact expectations for the two hit rates
            "top1": float((k / months).mean()),
            "top3": float(np.mean([1 - comb(months - int(ki), 3) / comb(months, 3) for ki in k])),
        },
    }
    return {
        "cutoff": str(cutoff),
        "data_end": str(data_end),
        "origins": origins,
        "months": months,
        "train_rows": hm.metrics["train_rows"],
        "monthly_probability": monthly,
        "placement": placement,
    }

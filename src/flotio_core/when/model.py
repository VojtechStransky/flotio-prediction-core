"""The hazard model: train() fits it, HazardModel.predict() gives each firm's monthly probabilities."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl

from .history import FEATURES, HORIZON, features_at, labels_at, month_index, purchase_months

INPUTS = FEATURES + ["h", "tm"]
NEG_RATE = 0.10
FIRST_ORIGIN = date(2016, 1, 1)


def _design(feats: pl.DataFrame, origin_mi: int, h: int) -> np.ndarray:
    x = feats.select(FEATURES).to_numpy().astype(np.float32)
    return np.column_stack([x, np.full(len(x), h, np.float32), np.full(len(x), (origin_mi + h) % 12 + 1, np.float32)])


@dataclass
class HazardModel:
    model: object
    data_end: date
    neg_rate: float = NEG_RATE
    horizon: int = HORIZON
    metrics: dict = field(default_factory=dict)

    def predict(self, pm: pl.DataFrame, origin: date) -> tuple[pl.Series, np.ndarray]:
        """Firms modelled at `origin` and their (n, horizon) monthly probabilities."""
        omi = month_index(origin)
        feats = features_at(pm, omi)
        if feats.height == 0:
            return feats["firm"], np.zeros((0, self.horizon))
        ph = np.zeros((feats.height, self.horizon))
        for h in range(self.horizon):
            q = self.model.predict_proba(_design(feats, omi, h))[:, 1]
            odds = q / np.maximum(1 - q, 1e-9) * self.neg_rate
            ph[:, h] = odds / (1 + odds)
        return feats["firm"], ph


def train(
    purchases: pl.DataFrame,
    data_end: date,
    *,
    first_origin: date = FIRST_ORIGIN,
    seed: int = 0,
    n_estimators: int = 400,
) -> HazardModel:
    """Fit on every month-start from `first_origin` using labels before `data_end`."""
    import xgboost as xgb

    pm = purchase_months(purchases)
    rng = np.random.default_rng(seed)
    end_mi = month_index(data_end)
    xs, ys = [], []
    for omi in range(month_index(first_origin), end_mi):
        feats = features_at(pm, omi)
        if feats.height == 0:
            continue
        y = labels_at(pm, feats["firm"], omi, end_mi)
        for h in range(HORIZON):
            if omi + h >= end_mi:
                break
            yh = y[:, h]
            keep = (yh == 1) | (rng.random(len(yh)) < NEG_RATE)
            if keep.any():
                xs.append(_design(feats.filter(pl.Series(keep)), omi, h))
                ys.append(yh[keep])
    if not xs:
        raise ValueError("no training rows — not enough purchase history before data_end")
    X, Y = np.concatenate(xs), np.concatenate(ys)
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        tree_method="hist",
        n_jobs=os.cpu_count() or 4,
        verbosity=0,
        random_state=seed,
    )
    model.fit(X, Y)
    return HazardModel(
        model=model, data_end=data_end, metrics={"train_rows": int(len(Y)), "train_positives": int(Y.sum())}
    )

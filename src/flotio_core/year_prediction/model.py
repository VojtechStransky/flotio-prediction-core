"""The 12-month classifier: fit() trains and calibrates it, WhoModel.predict() scores firms."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from .features import add_months, build_features, prepare_xy
from .threshold import threshold_at_max_recall_under_fpr


def train_model(X_train, y_train, X_val, y_val, seed: int = 42):
    """Gradient-boosted classifier; buyers are weighted up to balance the classes."""
    import xgboost as xgb

    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="auc",
        random_state=seed,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def calibrate_model(model, X_val, y_val) -> IsotonicRegression:
    """Map raw scores to observed frequencies on the validation table.

    Weighting buyers up (scale_pos_weight) inflates the raw probabilities. They
    still rank firms correctly, but a displayed "60 %" would be misleading.
    Isotonic regression keeps the order and fixes the level.
    """
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(model.predict_proba(X_val)[:, 1], y_val)
    return calibrator


@dataclass
class WhoModel:
    model: object
    calibrator: IsotonicRegression
    encoder: LabelEncoder
    feature_names: list[str]
    threshold: float
    metrics: dict = field(default_factory=dict)

    def predict(self, purchases: pl.DataFrame, cutoff: date) -> tuple[pl.Series, np.ndarray, np.ndarray]:
        """Firms known at `cutoff`, their calibrated 12-month probability and the yes/no call."""
        X, _, firms, _, _ = prepare_xy(build_features(purchases, cutoff), self.encoder)
        raw = self.model.predict_proba(X)[:, 1]
        # The threshold was chosen on raw scores, so the call uses them too.
        return firms, self.calibrator.predict(raw), raw >= self.threshold


def fit(purchases: pl.DataFrame, val_cutoff: date, max_fpr: float = 0.20, seed: int = 42) -> WhoModel:
    """Train on the cutoff a year before `val_cutoff`, validate and calibrate at `val_cutoff`.

    The 12 months after `val_cutoff` must be complete in `purchases`.
    """
    train_cutoff = add_months(val_cutoff, -12)
    X_train, y_train, _, feature_names, encoder = prepare_xy(build_features(purchases, train_cutoff))
    X_val, y_val, _, _, _ = prepare_xy(build_features(purchases, val_cutoff), encoder)

    model = train_model(X_train, y_train, X_val, y_val, seed)
    raw_val = model.predict_proba(X_val)[:, 1]
    threshold, operating_point = threshold_at_max_recall_under_fpr(y_val, raw_val, max_fpr)

    return WhoModel(
        model=model,
        calibrator=calibrate_model(model, X_val, y_val),
        encoder=encoder,
        feature_names=feature_names,
        threshold=threshold,
        metrics={
            "train_rows": int(len(y_train)),
            "train_buyers": int(y_train.sum()),
            "val_rows": int(len(y_val)),
            "val_buyers": int(y_val.sum()),
            "val_auc": float(roc_auc_score(y_val, raw_val)),
            "val_operating_point": operating_point,
        },
    )

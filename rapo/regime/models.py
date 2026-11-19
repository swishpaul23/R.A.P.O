"""Stress classifiers. Each factory returns a fresh, unfitted model."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ModelFactory = Callable[[], ClassifierMixin]


def logistic() -> Pipeline:
    """Regularized logistic regression on standardized features (the interpretable baseline)."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000)),
    ])


def boosting() -> ClassifierMixin:
    """Shallow, heavily regularized XGBoost: few crash examples, so depth and leaves stay small."""
    from xgboost import XGBClassifier  # dev-only dependency: training runs offline

    return XGBClassifier(
        max_depth=3, learning_rate=0.05, n_estimators=200, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, min_child_weight=5, eval_metric="logloss", random_state=0, n_jobs=1,
    )


def vix_rule(features: pd.DataFrame, threshold: float = 30.0) -> pd.Series:
    """The simple baseline: 'stress' whenever VIX is above `threshold`."""
    v = features["vix"]
    out = (v > threshold).astype(float)
    out[v.isna()] = np.nan
    return out.rename("vix_rule")

"""Walk-forward (expanding window) stress probabilities with no look-ahead.

At each refit date d, the model trains on rows dated at least `gap` rows before d
(labels look `gap` days ahead, so a later row's label isn't known yet at d) and
predicts rows d .. d + refit_every - 1. Probabilities are isotonic-calibrated
inside the training fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

from rapo.regime.models import ModelFactory


def _fit(factory: ModelFactory, X: pd.DataFrame, y: pd.Series, calibrate: bool):
    if calibrate:
        try:
            model = CalibratedClassifierCV(factory(), method="isotonic", cv=TimeSeriesSplit(3))
            return model.fit(X, y)
        except ValueError:
            pass  # a calibration split had one class; fall back to the raw model
    return factory().fit(X, y)


def walk_forward_proba(features: pd.DataFrame, labels: pd.Series, factory: ModelFactory,
                       min_train: int = 750, refit_every: int = 21, gap: int = 21,
                       calibrate: bool = True, return_log: bool = False):
    X_all = features.dropna()
    y_all = labels.reindex(X_all.index)
    proba = pd.Series(np.nan, index=features.index, name="proba")
    log: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    n = len(X_all)
    trainable = y_all.notna().to_numpy()
    d = None
    for i in range(gap, n):
        if trainable[: i - gap + 1].sum() >= min_train:
            d = i
            break
    if d is None:
        return (proba, log) if return_log else proba

    while d < n:
        train_idx = X_all.index[: d - gap + 1]
        mask = y_all.loc[train_idx].notna()
        X_tr, y_tr = X_all.loc[train_idx][mask], y_all.loc[train_idx][mask]
        block = X_all.index[d: d + refit_every]

        if y_tr.nunique() < 2:
            proba.loc[block] = float(y_tr.mean())
        else:
            model = _fit(factory, X_tr, y_tr, calibrate)
            proba.loc[block] = model.predict_proba(X_all.loc[block])[:, 1]
        log.append((X_tr.index.max(), block[0]))
        d += refit_every

    return (proba, log) if return_log else proba

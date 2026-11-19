import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from rapo.regime.features import build_features
from rapo.regime.labels import stress_label
from rapo.regime.models import boosting, logistic, vix_rule
from rapo.regime.walkforward import walk_forward_proba


class Recorder(BaseEstimator, ClassifierMixin):
    seen: list = []

    def fit(self, X, y):
        Recorder.seen.append(X.index.max())
        self.classes_ = np.array([0.0, 1.0])
        self.p_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.p_), np.full(len(X), self.p_)])


@pytest.fixture
def xy(regime_px):
    f = build_features(regime_px.drop(columns="^IRX")).drop(columns="curve")
    y = stress_label(regime_px["SPY"], horizon=21, threshold=-0.04)
    return f, y


def test_training_never_sees_the_future(xy):
    f, y = xy
    Recorder.seen = []
    proba, log = walk_forward_proba(f, y, Recorder, min_train=300, refit_every=50, gap=21,
                                    calibrate=False, return_log=True)
    assert len(log) > 3 and len(Recorder.seen) == len(log)
    pos = {d: i for i, d in enumerate(f.index)}
    for (train_end, predict_start), seen in zip(log, Recorder.seen):
        assert seen == train_end
        assert pos[train_end] <= pos[predict_start] - 21


def test_nan_before_min_train(xy):
    f, y = xy
    proba = walk_forward_proba(f, y, logistic, min_train=500, refit_every=63, gap=21)
    first = proba.first_valid_index()
    assert f.loc[:first].dropna().shape[0] >= 500 + 21
    assert proba.loc[:first].iloc[:-1].isna().all()


def test_single_class_fold_returns_base_rate(xy):
    f, _ = xy
    zeros = pd.Series(0.0, index=f.index)
    proba = walk_forward_proba(f, zeros, logistic, min_train=300, refit_every=100, gap=21)
    assert (proba.dropna() == 0.0).all()


@pytest.mark.parametrize("factory", [logistic, boosting])
def test_probabilities_in_unit_interval(xy, factory):
    f, y = xy
    proba = walk_forward_proba(f, y, factory, min_train=400, refit_every=126, gap=21)
    p = proba.dropna()
    assert len(p) > 0 and ((p >= 0) & (p <= 1)).all()


def test_vix_rule():
    f = pd.DataFrame({"vix": [12.0, 31.0, np.nan]}, index=pd.bdate_range("2024-01-01", periods=3))
    out = vix_rule(f, threshold=30)
    assert out.iloc[:2].tolist() == [0.0, 1.0] and np.isnan(out.iloc[2])

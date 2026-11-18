import numpy as np
import pandas as pd
import pytest

from rapo.regime.evaluate import block_bootstrap_ci, calibration_table, forecast_scores

IDX = pd.bdate_range("2020-01-01", periods=400)


def test_perfect_predictions():
    y = pd.Series(np.tile([0.0, 1.0], 200), index=IDX)
    s = forecast_scores(y, y)
    assert s["auc"] == 1.0 and s["brier"] == 0.0 and s["base_rate"] == 0.5 and s["n"] == 400


def test_constant_half_has_brier_quarter_and_ignores_nan():
    y = pd.Series(np.tile([0.0, 1.0], 200), index=IDX)
    p = pd.Series(0.5, index=IDX)
    p.iloc[:10] = np.nan
    s = forecast_scores(p, y)
    assert s["brier"] == pytest.approx(0.25) and s["n"] == 390


def test_calibration_table_bins():
    y = pd.Series(np.tile([0.0, 1.0], 200), index=IDX)
    p = pd.Series(np.linspace(0, 1, 400), index=IDX)
    t = calibration_table(p, y, bins=4)
    assert list(t.columns) == ["mean_pred", "observed", "count"]
    assert t["count"].sum() == 400 and len(t) == 4


def test_bootstrap_of_identical_series_is_zero():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 400), index=IDX)
    assert block_bootstrap_ci(r, r, lambda x: x.mean()) == (0.0, 0.0, 0.0)


def test_bootstrap_detects_a_clear_edge():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 400), index=IDX)
    point, lo, hi = block_bootstrap_ci(r + 0.001, r, lambda x: x.mean())
    assert point == pytest.approx(0.001) and lo > 0 and hi > lo


def test_bootstrap_is_reproducible():
    rng = np.random.default_rng(2)
    a = pd.Series(rng.normal(0, 0.01, 400), index=IDX)
    b = pd.Series(rng.normal(0, 0.01, 400), index=IDX)
    assert block_bootstrap_ci(a, b, lambda x: x.std(), seed=5) == block_bootstrap_ci(a, b, lambda x: x.std(), seed=5)

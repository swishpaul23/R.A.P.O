import numpy as np
import pandas as pd
import pytest

from rapo.metrics import (STRESS_WINDOWS, ann_vol, cagr, calmar, max_drawdown, sharpe,
                          sortino, summary, var_hist, window)

IDX = pd.bdate_range("2024-01-01", periods=252)


def test_cagr_doubling_over_252_days():
    r = pd.Series([2 ** (1 / 252) - 1] * 252, index=IDX)
    assert cagr(r) == pytest.approx(1.0)


def test_ann_vol_scales_by_sqrt_252():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 252), index=IDX)
    assert ann_vol(r) == pytest.approx(r.std() * np.sqrt(252))


def test_max_drawdown_known_path():
    r = pd.Series([0.10, -0.50, 0.20])  # wealth 1.1 -> 0.55 -> 0.66
    assert max_drawdown(r) == pytest.approx(-0.50)


def test_max_drawdown_counts_loss_from_start():
    assert max_drawdown(pd.Series([-0.2, 0.1])) == pytest.approx(-0.2)


def test_calmar_is_cagr_over_drawdown():
    r = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 252), index=IDX)
    assert calmar(r) == pytest.approx(cagr(r) / abs(max_drawdown(r)))


def test_sharpe_zero_when_excess_mean_is_zero():
    rf = pd.Series(0.0001, index=IDX)
    noise = np.random.default_rng(0).normal(0, 0.01, 252)
    r = rf + (noise - noise.mean())
    assert sharpe(r, rf) == pytest.approx(0.0, abs=1e-12)


def test_sharpe_aligns_sparse_rf_by_forward_fill():
    r = pd.Series(np.random.default_rng(2).normal(0.0005, 0.01, 252), index=IDX)
    rf_sparse = pd.Series([0.0001], index=[IDX[0]])
    assert sharpe(r, rf_sparse) == pytest.approx(sharpe(r, pd.Series(0.0001, index=IDX)))


def test_var_hist_is_5th_percentile():
    r = pd.Series(np.arange(-50, 50) / 1000)
    assert var_hist(r) == pytest.approx(np.percentile(r, 5))


def test_sortino_ignores_upside():
    rf = pd.Series(0.0, index=IDX)
    r = pd.Series(np.random.default_rng(3).normal(0, 0.01, 252), index=IDX)
    bigger_up = r.copy()
    bigger_up.iloc[r.argmax()] += 0.05
    assert sortino(bigger_up, rf) > sortino(r, rf)
    # Sharpe penalizes the extra upside volatility; Sortino does not.
    assert (sortino(bigger_up, rf) / sortino(r, rf)) > (sharpe(bigger_up, rf) / sharpe(r, rf))


def test_window_slices_dates():
    r = pd.Series(0.0, index=pd.bdate_range("2019-06-01", "2023-06-01"))
    w = window(r, "2022 rate shock")
    assert w.index.min() >= pd.Timestamp("2022-01-01")
    assert w.index.max() <= pd.Timestamp("2022-12-31")
    assert set(STRESS_WINDOWS) == {"2020 COVID crash", "2022 rate shock"}


def test_summary_has_all_fields():
    r = pd.Series(np.random.default_rng(4).normal(0.0005, 0.01, 252), index=IDX)
    turnover = pd.Series([1.0, 0.5], index=[IDX[0], IDX[100]])
    s = summary(r, pd.Series(0.0, index=IDX), turnover)
    assert {"cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar",
            "var_95", "annual_turnover"} <= set(s.index)
    assert s["annual_turnover"] == pytest.approx(1.5)

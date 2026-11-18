import numpy as np
import pandas as pd
import pytest

from rapo.backtest import BacktestResult, eligible_assets, run_backtest
from rapo.strategies import equal_weight


def frame(rows, cols=("A", "B")):
    return pd.DataFrame(rows, columns=list(cols), index=pd.bdate_range("2024-01-01", periods=len(rows)))


HAND = frame([[0.01, 0.01], [0.02, -0.01], [0.10, 0.00], [0.00, 0.10]])


def test_two_asset_hand_example():
    res = run_backtest(HAND, equal_weight, lookback=2, rebalance_every=100, cost_bps=0)
    assert isinstance(res, BacktestResult)
    # Day 1: 0.5 * 10% = 5%. Drift -> wB = 0.5 / 1.05. Day 2: wB * 10%.
    assert res.returns.tolist() == pytest.approx([0.05, 0.5 / 1.05 * 0.10])


def test_costs_charged_on_turnover():
    res = run_backtest(HAND, equal_weight, lookback=2, rebalance_every=100, cost_bps=10)
    assert res.turnover.iloc[0] == pytest.approx(1.0)  # from all cash to fully invested
    assert res.returns.iloc[0] == pytest.approx(0.05 - 0.001)


def test_no_lookahead():
    rets = frame(np.random.default_rng(0).normal(0, 0.01, size=(80, 2)))
    seen = []

    def spy(window):
        seen.append(window.index.max())
        return equal_weight(window)

    res = run_backtest(rets, spy, lookback=20, rebalance_every=5)
    assert len(seen) == len(res.weights)
    for max_seen, rebal_date in zip(seen, res.weights.index):
        assert max_seen < rebal_date


def test_new_asset_enters_only_after_full_window():
    data = np.random.default_rng(1).normal(0, 0.01, size=(80, 2))
    data[:30, 1] = np.nan  # NEW has no returns before row 30
    rets = frame(data, cols=("OLD", "NEW"))
    res = run_backtest(rets, equal_weight, lookback=20, rebalance_every=5)
    row_of = {d: rets.index.get_loc(d) for d in res.weights.index}
    for date, w in res.weights["NEW"].items():
        if row_of[date] < 50:
            assert w == 0
        else:
            assert w > 0


def test_eligible_assets_requires_full_window_and_today():
    data = np.ones((30, 2)) * 0.01
    data[:10, 1] = np.nan
    rets = frame(data)
    assert eligible_assets(rets, 25, 20) == ["A"]
    assert eligible_assets(rets, 29, 19) == ["A", "B"]


def test_single_eligible_asset_gets_full_weight():
    data = np.random.default_rng(2).normal(0, 0.01, size=(40, 2))
    data[:, 1] = np.nan
    res = run_backtest(frame(data), equal_weight, lookback=10, rebalance_every=10)
    assert (res.weights["A"] == 1.0).all()
    assert (res.weights["B"] == 0.0).all()


def test_zero_eligible_holds_cash():
    data = np.random.default_rng(3).normal(0, 0.01, size=(60, 2))
    data[:30, :] = np.nan
    rets = frame(data)
    res = run_backtest(rets, equal_weight, lookback=20, rebalance_every=5, cost_bps=10)
    cash_days = res.returns.loc[: rets.index[49]]
    assert (cash_days == 0).all()
    assert (res.costs.loc[: rets.index[49]] == 0).all()
    assert res.weights.iloc[-1].sum() == pytest.approx(1.0)


def test_too_short_raises():
    rets = frame(np.zeros((100, 2)))
    with pytest.raises(ValueError, match="253"):
        run_backtest(rets, equal_weight, lookback=252)

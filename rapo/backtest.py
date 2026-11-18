"""Walk-forward backtest engine.

At each rebalance day t the strategy sees only returns strictly before t
(rows t-lookback .. t-1), so no future data can leak into a decision.
Between rebalances the weights drift with prices, exactly as a real
buy-and-hold position would. Trading costs are charged on turnover.

Portfolio return each day is sum(w_i * r_i) with SIMPLE returns. (Summing
weighted log returns, which the v1 code did, is not the portfolio's return.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rapo.strategies import Strategy


@dataclass
class BacktestResult:
    returns: pd.Series      # daily net return, from the first rebalance day
    weights: pd.DataFrame   # target weights at each rebalance date (all tickers)
    turnover: pd.Series     # sum |target - drifted| at each rebalance date
    costs: pd.Series        # cost charged at each rebalance date (fraction of NAV)


def eligible_assets(returns: pd.DataFrame, t: int, lookback: int) -> list[str]:
    """Tickers with a complete lookback window before row t and a return at row t."""
    window = returns.iloc[t - lookback:t]
    has_window = window.notna().all()
    has_today = returns.iloc[t].notna()
    return [c for c in returns.columns if has_window[c] and has_today[c]]


def run_backtest(returns: pd.DataFrame, strategy: Strategy, lookback: int = 252,
                 rebalance_every: int = 63, cost_bps: float = 10.0) -> BacktestResult:
    n = len(returns)
    if n <= lookback:
        raise ValueError(
            f"Need at least {lookback + 1} days of returns for a {lookback}-day lookback; got {n}."
        )

    cols = list(returns.columns)
    rets = returns.to_numpy(dtype=float)
    w = np.zeros(len(cols))
    daily = np.zeros(n - lookback)
    rebal_dates, targets, turnovers, costs = [], [], [], []

    for i, t in enumerate(range(lookback, n)):
        cost = 0.0
        if (t - lookback) % rebalance_every == 0:
            names = eligible_assets(returns, t, lookback)
            target = np.zeros(len(cols))
            if names:
                weights = strategy(returns.iloc[t - lookback:t][names])
                target = weights.reindex(cols).fillna(0.0).to_numpy()
            turnover = float(np.abs(target - w).sum())
            cost = cost_bps / 1e4 * turnover
            w = target
            rebal_dates.append(returns.index[t])
            targets.append(target)
            turnovers.append(turnover)
            costs.append(cost)

        r = np.nan_to_num(rets[t], nan=0.0)  # a held asset with no quote today counts as flat
        gross = float(w @ r)
        daily[i] = gross - cost
        if w.sum() > 0:
            w = w * (1.0 + r) / (1.0 + gross)

    idx = pd.DatetimeIndex(rebal_dates)
    return BacktestResult(
        returns=pd.Series(daily, index=returns.index[lookback:], name="returns"),
        weights=pd.DataFrame(targets, index=idx, columns=cols),
        turnover=pd.Series(turnovers, index=idx, name="turnover"),
        costs=pd.Series(costs, index=idx, name="costs"),
    )

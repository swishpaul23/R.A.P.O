"""Allocation strategies.

Every strategy has the same shape: a DataFrame of daily simple returns
(lookback window x eligible tickers, no NaN) in, a weight Series indexed by
the same tickers out. Weights are long-only and sum to 1.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from rapo.estimators import annual_mean, ledoit_wolf_cov

Strategy = Callable[[pd.DataFrame], pd.Series]


def _clean(w: np.ndarray, index: pd.Index) -> pd.Series:
    """Clip optimizer noise below zero and renormalize to sum to 1."""
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    total = w.sum()
    w = np.full(len(w), 1.0 / len(w)) if total <= 0 else w / total
    return pd.Series(w, index=index)


def equal_weight(returns: pd.DataFrame) -> pd.Series:
    n = returns.shape[1]
    return pd.Series(np.full(n, 1.0 / n), index=returns.columns)


def inverse_vol(returns: pd.DataFrame) -> pd.Series:
    """Weight each asset by 1 / its volatility."""
    vol = returns.std().to_numpy()
    vol = np.where(vol > 0, vol, np.nan)
    inv = np.nan_to_num(1.0 / vol, nan=0.0)
    return _clean(inv, returns.columns)


def _solve(objective, n: int, upper: float) -> np.ndarray | None:
    res = minimize(
        objective,
        x0=np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, upper)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return res.x if res.success else None


def min_variance(returns: pd.DataFrame) -> pd.Series:
    """Lowest-variance long-only portfolio under Ledoit-Wolf covariance."""
    cov = ledoit_wolf_cov(returns).to_numpy()
    n = cov.shape[0]
    w = _solve(lambda w: w @ cov @ w, n, 1.0)
    return _clean(w if w is not None else np.full(n, 1.0 / n), returns.columns)


def make_max_sharpe(rf_annual: float = 0.0, cap: float = 0.40) -> Strategy:
    """Max-Sharpe strategy with per-asset cap; falls back to min variance.

    The fallback triggers when no asset's expected return beats the risk-free
    rate (the Sharpe objective has no sensible maximum) or the solver fails.
    """

    def max_sharpe(returns: pd.DataFrame) -> pd.Series:
        mu = annual_mean(returns).to_numpy()
        if (mu <= rf_annual).all():
            return min_variance(returns)
        cov = ledoit_wolf_cov(returns).to_numpy()
        n = len(mu)
        upper = max(cap, 1.0 / n)

        def neg_sharpe(w):
            vol = np.sqrt(max(w @ cov @ w, 1e-18))
            return -(w @ mu - rf_annual) / vol

        w = _solve(neg_sharpe, n, upper)
        if w is None:
            return min_variance(returns)
        return _clean(w, returns.columns)

    return max_sharpe


def fixed_weights(target: dict[str, float]) -> Strategy:
    """Constant target weights (e.g. 60/40), renormalized over the eligible assets.

    Assets outside the target get 0. If none of the target assets is eligible yet,
    falls back to equal weight so the portfolio is never empty.
    """

    def strategy(returns: pd.DataFrame) -> pd.Series:
        w = pd.Series({c: float(target.get(c, 0.0)) for c in returns.columns})
        if w.sum() <= 0:
            return equal_weight(returns)
        return w / w.sum()

    return strategy

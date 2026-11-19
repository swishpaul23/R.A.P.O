"""Performance metrics on daily simple returns (252 trading days per year)."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

STRESS_WINDOWS: dict[str, tuple[str, str]] = {
    "2020 COVID crash": ("2020-02-01", "2020-06-30"),
    "2022 rate shock": ("2022-01-01", "2022-12-31"),
}


def _excess(r: pd.Series, rf: pd.Series) -> pd.Series:
    """r minus the risk-free rate, with rf forward-filled onto r's dates."""
    rf_aligned = rf.reindex(rf.index.union(r.index)).sort_index().ffill().bfill().reindex(r.index)
    return r - rf_aligned.fillna(0.0)


def total_return(r: pd.Series) -> float:
    """Compounded gain over the whole series, not annualized."""
    return float((1.0 + r).prod() - 1.0)


def cagr(r: pd.Series) -> float:
    growth = float((1.0 + r).prod())
    years = len(r) / TRADING_DAYS
    return growth ** (1.0 / years) - 1.0 if years > 0 and growth > 0 else float("nan")


def ann_vol(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(TRADING_DAYS))


def sharpe(r: pd.Series, rf: pd.Series) -> float:
    ex = _excess(r, rf)
    sd = ex.std()
    return float(ex.mean() * TRADING_DAYS / (sd * np.sqrt(TRADING_DAYS))) if sd > 1e-12 else float("nan")


def sortino(r: pd.Series, rf: pd.Series) -> float:
    """Like Sharpe, but only downside deviation counts as risk."""
    ex = _excess(r, rf)
    downside = np.sqrt((np.minimum(ex, 0.0) ** 2).mean())
    if downside <= 0:
        return float("nan")
    return float(ex.mean() * TRADING_DAYS / (downside * np.sqrt(TRADING_DAYS)))


def max_drawdown(r: pd.Series) -> float:
    """Worst peak-to-trough loss, as a negative fraction. Starts from wealth 1.0."""
    wealth = np.concatenate([[1.0], (1.0 + r).cumprod().to_numpy()])
    peak = np.maximum.accumulate(wealth)
    return float((wealth / peak - 1.0).min())


def calmar(r: pd.Series) -> float:
    mdd = max_drawdown(r)
    return cagr(r) / abs(mdd) if mdd < 0 else float("nan")


def var_hist(r: pd.Series, level: float = 0.95) -> float:
    """Historical daily Value at Risk: the (1 - level) percentile return (negative)."""
    return float(np.percentile(r.dropna(), (1.0 - level) * 100))


def summary(r: pd.Series, rf: pd.Series, turnover: pd.Series | None = None) -> pd.Series:
    years = len(r) / TRADING_DAYS
    return pd.Series({
        "total_return": total_return(r),
        "cagr": cagr(r),
        "ann_vol": ann_vol(r),
        "sharpe": sharpe(r, rf),
        "sortino": sortino(r, rf),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar(r),
        "var_95": var_hist(r),
        "annual_turnover": float(turnover.sum() / years) if turnover is not None and years > 0 else float("nan"),
    })


def window(r: pd.Series, name: str) -> pd.Series:
    start, end = STRESS_WINDOWS[name]
    return r.loc[start:end]


def rebound_capture(r: pd.Series, bench: pd.Series, trough: str, days: int = 63) -> float:
    """Share of the benchmark's rebound the strategy captured in the `days` after `trough`.

    1.0 = matched the recovery; 0.5 = got half of it. NaN if the benchmark didn't rise.
    """
    after_r = r.loc[pd.Timestamp(trough):].iloc[1: days + 1]
    after_b = bench.loc[pd.Timestamp(trough):].iloc[1: days + 1]
    gain_b = float((1.0 + after_b).prod() - 1.0)
    if gain_b <= 0:
        return float("nan")
    return float((1.0 + after_r).prod() - 1.0) / gain_b


def whipsaw_count(tilt: pd.Series, weeks: int = 4, threshold: float = 0.10) -> int:
    """Number of defensive moves (tilt up by >= threshold) reversed within `weeks`.

    Each one is a round trip that cost trading fees without lasting protection.
    """
    window = weeks * 5
    values = tilt.fillna(0.0).to_numpy()
    count, low, rising, start, peak = 0, values[0] if len(values) else 0.0, False, 0, 0.0
    for i, v in enumerate(values):
        if not rising:
            if v - low >= threshold - 1e-12:
                rising, start, peak = True, i, v
            else:
                low = min(low, v)
            continue
        peak = max(peak, v)
        if peak - v >= threshold - 1e-12:
            if i - start <= window:
                count += 1
            rising, low = False, v
        elif i - start > window:
            rising, low = False, v
    return count

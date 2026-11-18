"""The Phase 2 experiment: do stress forecasts make HRP / equal weight hold up better?

Everything here is out-of-sample: probabilities come from walk-forward models, and
strategy results are reported only from the first date a probability exists.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from rapo.backtest import run_backtest
from rapo.hrp import hrp
from rapo.metrics import (STRESS_WINDOWS, max_drawdown, rebound_capture, summary, whipsaw_count,
                          window)
from rapo.regime.evaluate import block_bootstrap_ci, calibration_table, forecast_scores
from rapo.regime.features import build_features
from rapo.regime.hmm import hmm_stress_proba
from rapo.regime.labels import stress_label
from rapo.regime.models import boosting, logistic, vix_rule
from rapo.regime.overlay import DEFAULT_DEFENSIVE, make_regime_overlay, tilt_from_proba
from rapo.regime.walkforward import walk_forward_proba
from rapo.returns import simple_returns
from rapo.strategies import Strategy, equal_weight, fixed_weights

MODELS = ["vix_rule", "hmm", "logistic", "boosting"]  # simplest first: ties go to the simpler model
TROUGHS = {"2020 COVID crash": "2020-03-23", "2022 rate shock": "2022-10-12"}


def _restrict(strategy: Strategy, assets: list[str]) -> Strategy:
    def wrapped(win: pd.DataFrame) -> pd.Series:
        cols = [c for c in win.columns if c in assets] or list(win.columns)
        w = strategy(win[cols]).reindex(win.columns).fillna(0.0)
        return w / w.sum()
    return wrapped


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, pd.Timestamp):
        return x.date().isoformat()
    return x


def _sharpe(excess: pd.Series) -> float:
    sd = excess.std()
    return float(excess.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")


def prepare_prices(prices: pd.DataFrame, start: str) -> pd.DataFrame:
    """Stock-market trading days from `start`, with index levels (^VIX, ^TNX) forward-filled."""
    prices = prices.loc[pd.Timestamp(start):]
    prices = prices[prices["SPY"].notna()].copy()
    levels = [c for c in prices.columns if c.startswith("^")]
    prices[levels] = prices[levels].ffill()
    return prices


def compute_probas(prices: pd.DataFrame, min_train: int = 750, refit_every: int = 21,
                   hmm_refit_every: int = 63) -> tuple[dict[str, pd.Series], pd.DataFrame, pd.Series]:
    """Walk-forward stress probabilities for every model, plus the features and labels used."""
    feats = build_features(prices)
    feats = feats.loc[:, feats.notna().any()]          # e.g. no ^IRX -> drop 'curve'
    labels = stress_label(prices["SPY"])
    spy_r = prices["SPY"].pct_change(fill_method=None)
    probas = {
        "vix_rule": vix_rule(feats),
        "hmm": hmm_stress_proba(spy_r, min_train=min_train, refit_every=hmm_refit_every),
        "logistic": walk_forward_proba(feats, labels, logistic, min_train, refit_every),
        "boosting": walk_forward_proba(feats, labels, boosting, min_train, refit_every),
    }
    return probas, feats, labels


def run_study(prices: pd.DataFrame, rf: pd.Series, universe: list[str], start: str,
              min_train: int = 750, refit_every: int = 21, hmm_refit_every: int = 63,
              lookback: int = 252, cost_bps: float = 10.0, bootstrap_n: int = 1000,
              precomputed: tuple | None = None) -> dict:
    prices = prepare_prices(prices, start)

    # 1. Forecasts.
    probas, feats, labels = precomputed or compute_probas(prices, min_train, refit_every, hmm_refit_every)
    common = pd.concat(probas, axis=1).join(labels).dropna().index
    # Climatology: always forecast the historical stress rate known at the time (labels end 21 days back).
    climatology = labels.expanding().mean().shift(21)
    reference = forecast_scores(climatology.loc[common], labels.loc[common])
    forecast = {m: forecast_scores(probas[m].loc[common], labels.loc[common]) for m in MODELS}
    for m in MODELS:
        forecast[m]["brier_skill"] = 1.0 - forecast[m]["brier"] / reference["brier"]
    calibration = {m: calibration_table(probas[m].loc[common], labels.loc[common]).to_dict("records")
                   for m in MODELS}
    chosen = min(MODELS, key=lambda m: (round(forecast[m]["brier"], 9), MODELS.index(m)))
    proba = probas[chosen]
    tilt = tilt_from_proba(proba)

    # 2. Backtests (rebalance monthly so the overlay can react; plain versions too, for a fair fight).
    cols = list(dict.fromkeys(universe + list(DEFAULT_DEFENSIVE) + ["SPY", "AGG"]))
    R = simple_returns(prices[[c for c in cols if c in prices.columns]])
    bt = dict(lookback=lookback, cost_bps=cost_bps)
    runs = {
        "Equal weight": run_backtest(R, _restrict(equal_weight, universe), rebalance_every=21, **bt),
        "HRP": run_backtest(R, _restrict(hrp, universe), rebalance_every=21, **bt),
        "Equal weight + regime": run_backtest(R, make_regime_overlay(equal_weight, proba, base_assets=universe),
                                              rebalance_every=21, **bt),
        "HRP + regime": run_backtest(R, make_regime_overlay(hrp, proba, base_assets=universe),
                                     rebalance_every=21, **bt),
        "60/40": run_backtest(R, fixed_weights({"SPY": 0.6, "AGG": 0.4}), rebalance_every=63, **bt),
    }
    oos_start = proba.first_valid_index()
    rets = {n: r.returns.loc[oos_start:] for n, r in runs.items()}
    rets["SPY"] = R["SPY"].loc[rets["HRP"].index]
    turn = {n: r.turnover.loc[oos_start:] for n, r in runs.items()}

    def row(name: str, r: pd.Series, t: pd.Series | None) -> dict:
        out = {"name": name, **summary(r, rf, t).to_dict()}
        for wname, trough in TROUGHS.items():
            if r.index[0] <= pd.Timestamp(trough) <= r.index[-1]:
                out[f"rebound_{wname}"] = rebound_capture(r, rets["SPY"], trough)
        if name.endswith("+ regime"):
            out["whipsaws"] = whipsaw_count(tilt.loc[oos_start:])
        return out

    order = ["SPY", "60/40", "Equal weight", "Equal weight + regime", "HRP", "HRP + regime"]
    backtests = [row(n, rets[n], turn.get(n)) for n in order]
    stress = {}
    for wname in STRESS_WINDOWS:
        sliced = {n: window(rets[n], wname) for n in order}
        if all(len(s) > 1 for s in sliced.values()):
            stress[wname] = [{"name": n, **summary(s, rf).to_dict()} for n, s in sliced.items()]

    # 3. Is the difference more than luck?
    rf_al = rf.reindex(rf.index.union(rets["HRP"].index)).ffill().bfill().reindex(rets["HRP"].index)
    boot = []
    for base in ("Equal weight", "HRP"):
        a, b = rets[f"{base} + regime"], rets[base]
        for metric, fn, x, y in (("sharpe", _sharpe, a - rf_al, b - rf_al),
                                 ("max_drawdown", max_drawdown, a, b)):
            diff, lo, hi = block_bootstrap_ci(x, y, fn, n=bootstrap_n)
            boot.append({"base": base, "metric": metric, "diff": diff, "lo": lo, "hi": hi,
                         "significant": bool(lo > 0 or hi < 0)})

    # 4. Series for the cycle chart.
    idx = rets["HRP"].index
    wealth = lambda r: (1.0 + r).cumprod()  # noqa: E731
    last = proba.dropna().index[-1]
    return _clean({
        "as_of": prices.index[-1],
        "chosen_model": chosen,
        "settings": {"universe": universe, "start": start, "lookback": lookback, "rebalance_every": 21,
                     "cost_bps": cost_bps, "min_train": min_train, "refit_every": refit_every,
                     "label": "SPY falls 8% or more within the next 21 trading days",
                     "defensive": DEFAULT_DEFENSIVE, "tilt": "0.40 x clip((EMA5(p) - 0.20) / 0.80, 0, 1)",
                     "oos_start": oos_start},
        "forecast": forecast,
        "forecast_reference": reference,
        "calibration": calibration,
        "backtests": backtests,
        "stress_windows": stress,
        "bootstrap": boot,
        "series": {"dates": [d.date().isoformat() for d in idx],
                   "proba": proba.reindex(idx).round(4).tolist(),
                   "tilt": tilt.reindex(idx).round(4).tolist(),
                   "spy": wealth(rets["SPY"]).round(5).tolist(),
                   "hrp": wealth(rets["HRP"]).round(5).tolist(),
                   "hrp_regime": wealth(rets["HRP + regime"]).round(5).tolist()},
        "current": {"date": last, "proba": float(proba.loc[last]), "tilt": float(tilt.loc[last])},
    })

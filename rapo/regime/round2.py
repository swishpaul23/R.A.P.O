"""Phase 2, round 2: choose a tilt rule on 2010-2019, then test it once on 2020+.

Protocol: docs/results/phase2-round2-protocol.md (committed before the first run).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rapo.backtest import run_backtest
from rapo.hrp import hrp
from rapo.metrics import STRESS_WINDOWS, max_drawdown, summary, whipsaw_count, window
from rapo.regime.evaluate import block_bootstrap_ci
from rapo.regime.overlay import DEFAULT_DEFENSIVE, make_tilt_overlay, percentile_tilt
from rapo.regime.study import _clean, _restrict, _sharpe
from rapo.returns import simple_returns
from rapo.strategies import equal_weight, fixed_weights

CANDIDATE_SIGNALS = ["logistic", "hmm", "vol_21"]  # protocol order (used for ties)
QS = (0.80, 0.90)
BASES = {"Equal weight": equal_weight, "HRP": hrp}


def run_round2(prices: pd.DataFrame, rf: pd.Series, universe: list[str], signals: dict[str, pd.Series],
               split: str = "2020-01-01", qs: tuple[float, ...] = QS, lookback: int = 252,
               min_history: int = 252, cost_bps: float = 10.0, bootstrap_n: int = 1000) -> dict:
    prices = prices[prices["SPY"].notna()]
    cols = list(dict.fromkeys(universe + list(DEFAULT_DEFENSIVE) + ["SPY", "AGG"]))
    R = simple_returns(prices[[c for c in cols if c in prices.columns]])
    bt = dict(lookback=lookback, cost_bps=cost_bps, rebalance_every=21)

    def rf_on(idx):
        return rf.reindex(rf.index.union(idx)).ffill().bfill().reindex(idx)

    plain = {n: run_backtest(R, _restrict(s, universe), **bt) for n, s in BASES.items()}
    start = max([plain["HRP"].returns.index[0]] + [s.first_valid_index() for s in signals.values()])
    split_ts = pd.Timestamp(split)
    dev = lambda r: r.loc[start: split_ts - pd.Timedelta(days=1)]  # noqa: E731
    hold = lambda r: r.loc[split_ts:]  # noqa: E731

    # 1. Development: score every candidate on 2010-2019 only.
    candidates, tilts, overlays = [], {}, {}
    for sig in CANDIDATE_SIGNALS:
        for q in qs:
            tilt = percentile_tilt(signals[sig], q, min_history=min_history)
            runs = {n: run_backtest(R, make_tilt_overlay(s, tilt, base_assets=universe), **bt)
                    for n, s in BASES.items()}
            deltas = {}
            for n in BASES:
                a, b = dev(runs[n].returns), dev(plain[n].returns)
                deltas[n] = _sharpe(a - rf_on(a.index)) - _sharpe(b - rf_on(b.index))
            candidates.append({"signal": sig, "q": q, "score": float(np.mean(list(deltas.values()))),
                               "sharpe_gain": deltas,
                               "avg_tilt": float(dev(tilt).mean()),
                               "whipsaws": whipsaw_count(dev(tilt))})
            tilts[(sig, q)], overlays[(sig, q)] = tilt, runs
    order = {(s, q): i for i, (s, q) in enumerate((s, q) for s in CANDIDATE_SIGNALS for q in sorted(qs, reverse=True))}
    best = max(candidates, key=lambda c: (round(c["score"], 9), -order[(c["signal"], c["q"])]))
    key = (best["signal"], best["q"])
    tilt, runs = tilts[key], overlays[key]

    # 2. Holdout: the locked rule, run once.
    sixty = run_backtest(R, fixed_weights({"SPY": 0.6, "AGG": 0.4}), lookback=lookback,
                         cost_bps=cost_bps, rebalance_every=63)
    rets = {"Equal weight": plain["Equal weight"].returns, "Equal weight + regime": runs["Equal weight"].returns,
            "HRP": plain["HRP"].returns, "HRP + regime": runs["HRP"].returns, "60/40": sixty.returns}
    turns = {"Equal weight": plain["Equal weight"].turnover, "Equal weight + regime": runs["Equal weight"].turnover,
             "HRP": plain["HRP"].turnover, "HRP + regime": runs["HRP"].turnover, "60/40": sixty.turnover}
    rets["SPY"] = R["SPY"].reindex(rets["HRP"].index)
    names = ["SPY", "60/40", "Equal weight", "Equal weight + regime", "HRP", "HRP + regime"]
    table = [{"name": n, **summary(hold(rets[n]), rf, hold(turns[n]) if n in turns else None).to_dict()}
             for n in names]
    stress = {}
    for wname in STRESS_WINDOWS:
        sliced = {n: window(hold(rets[n]), wname) for n in names}
        if all(len(s) > 1 for s in sliced.values()):
            stress[wname] = [{"name": n, **summary(s, rf).to_dict()} for n, s in sliced.items()]
    boot = []
    for base in BASES:
        a, b = hold(rets[f"{base} + regime"]), hold(rets[base])
        for metric, fn, x, y in (("sharpe", _sharpe, a - rf_on(a.index), b - rf_on(b.index)),
                                 ("max_drawdown", max_drawdown, a, b)):
            diff, lo, hi = block_bootstrap_ci(x, y, fn, n=bootstrap_n)
            boot.append({"base": base, "metric": metric, "diff": diff, "lo": lo, "hi": hi,
                         "significant": bool(lo > 0 or hi < 0)})

    idx = hold(rets["HRP"]).index
    wealth = lambda r: (1.0 + r).cumprod()  # noqa: E731
    return _clean({
        "periods": {"development_start": start, "split": split, "holdout_start": idx[0] if len(idx) else None,
                    "holdout_end": idx[-1] if len(idx) else None},
        "development": candidates,
        "chosen": {"signal": best["signal"], "q": best["q"]},
        "holdout": {"table": table, "stress_windows": stress, "bootstrap": boot,
                    "whipsaws": whipsaw_count(hold(tilt)), "avg_tilt": float(hold(tilt).mean())},
        "series": {"dates": [d.date().isoformat() for d in idx],
                   "signal": signals[best["signal"]].reindex(idx).round(5).tolist(),
                   "tilt": tilt.reindex(idx).round(4).tolist(),
                   "spy": wealth(hold(rets["SPY"])).round(5).tolist(),
                   "hrp": wealth(hold(rets["HRP"])).round(5).tolist(),
                   "hrp_regime": wealth(hold(rets["HRP + regime"])).round(5).tolist()},
        "current": {"date": tilt.index[-1], "tilt": float(tilt.iloc[-1])},
    })

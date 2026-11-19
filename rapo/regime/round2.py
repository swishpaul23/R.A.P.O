"""Phase 2, round 2: choose a tilt rule on 2010-2019, then test it once on 2020+.

Protocol: docs/results/phase2-round2-protocol.md (committed before the first run).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rapo.backtest import run_backtest
from rapo.hrp import hrp
from rapo.metrics import STRESS_WINDOWS, max_drawdown, summary, whipsaw_count
from rapo.regime.evaluate import block_bootstrap_ci
from rapo.regime.overlay import DEFAULT_DEFENSIVE, make_tilt_overlay, percentile_tilt
from rapo.regime.study import _clean, _restrict, _sharpe
from rapo.returns import simple_returns
from rapo.strategies import equal_weight, fixed_weights

CANDIDATE_SIGNALS = ["logistic", "hmm", "vol_21"]  # protocol order (used for ties)
QS = (0.80, 0.90)
BASES = {"Equal weight": equal_weight, "HRP": hrp}


def evaluate_rule(R: pd.DataFrame, rf: pd.Series, universe: list[str], tilt: pd.Series,
                  start: pd.Timestamp, end: pd.Timestamp | None, windows: dict[str, tuple[str, str]],
                  defensive: dict[str, float], sixty: dict[str, float], lookback: int, cost_bps: float,
                  bootstrap_n: int, signal: pd.Series | None = None) -> dict:
    """Run a tilt rule and its plain bases, then report one test period: table, windows, bootstrap, series."""
    bt = dict(lookback=lookback, cost_bps=cost_bps, rebalance_every=21)
    runs = {}
    for name, s in BASES.items():
        runs[name] = run_backtest(R, _restrict(s, universe), **bt)
        runs[f"{name} + regime"] = run_backtest(
            R, make_tilt_overlay(s, tilt, defensive=defensive, base_assets=universe), **bt)
    runs["60/40"] = run_backtest(R, fixed_weights(sixty), lookback=lookback, cost_bps=cost_bps, rebalance_every=63)
    test = lambda r: r.loc[start:end]  # noqa: E731
    rets = {n: test(r.returns) for n, r in runs.items()}
    rets["SPY"] = test(R["SPY"]).reindex(rets["HRP"].index)
    names = ["SPY", "60/40", "Equal weight", "Equal weight + regime", "HRP", "HRP + regime"]
    table = [{"name": n, **summary(rets[n], rf, test(runs[n].turnover) if n in runs else None).to_dict()}
             for n in names]
    stress = {}
    for wname, (ws, we) in windows.items():
        sliced = {n: rets[n].loc[ws:we] for n in names}
        if all(len(s) > 1 for s in sliced.values()):
            stress[wname] = [{"name": n, **summary(s, rf).to_dict()} for n, s in sliced.items()]

    def rf_on(idx):
        return rf.reindex(rf.index.union(idx)).ffill().bfill().reindex(idx)

    boot = []
    for base in BASES:
        a, b = rets[f"{base} + regime"], rets[base]
        for metric, fn, x, y in (("sharpe", _sharpe, a - rf_on(a.index), b - rf_on(b.index)),
                                 ("max_drawdown", max_drawdown, a, b)):
            diff, lo, hi = block_bootstrap_ci(x, y, fn, n=bootstrap_n)
            boot.append({"base": base, "metric": metric, "diff": diff, "lo": lo, "hi": hi,
                         "significant": bool(lo > 0 or hi < 0)})
    idx = rets["HRP"].index
    wealth = lambda r: (1.0 + r).cumprod()  # noqa: E731
    series = {"dates": [d.date().isoformat() for d in idx],
              "tilt": tilt.reindex(idx).round(4).tolist(),
              "spy": wealth(rets["SPY"]).round(5).tolist(),
              "hrp": wealth(rets["HRP"]).round(5).tolist(),
              "hrp_regime": wealth(rets["HRP + regime"]).round(5).tolist()}
    if signal is not None:
        series["signal"] = signal.reindex(idx).round(5).tolist()
    return {"table": table, "stress_windows": stress, "bootstrap": boot,
            "whipsaws": whipsaw_count(test(tilt)), "avg_tilt": float(test(tilt).mean()), "series": series}


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
    holdout = evaluate_rule(R, rf, universe, tilt, split_ts, None, STRESS_WINDOWS, DEFAULT_DEFENSIVE,
                            {"SPY": 0.6, "AGG": 0.4}, lookback, cost_bps, bootstrap_n,
                            signal=signals[best["signal"]])
    return _clean({
        "periods": {"development_start": start, "split": split,
                    "holdout_start": holdout["series"]["dates"][0] if holdout["series"]["dates"] else None,
                    "holdout_end": holdout["series"]["dates"][-1] if holdout["series"]["dates"] else None},
        "development": candidates,
        "chosen": {"signal": best["signal"], "q": best["q"]},
        "holdout": {k: holdout[k] for k in ("table", "stress_windows", "bootstrap", "whipsaws", "avg_tilt")},
        "series": holdout["series"],
        "current": {"date": tilt.index[-1], "tilt": float(tilt.iloc[-1])},
    })

"""Backward test: the rule locked in round 2, run once on 2000-2009.

Protocol: docs/results/phase2-backward-protocol.md (committed before the run).
Uses long-history fund stand-ins because most of today's ETFs didn't exist yet.
"""

from __future__ import annotations

import pandas as pd

from rapo.regime.hmm import hmm_stress_proba
from rapo.regime.overlay import percentile_tilt
from rapo.regime.round2 import evaluate_rule
from rapo.regime.study import _clean
from rapo.returns import simple_returns

LOCKED_RULE = {"signal": "hmm", "q": 0.9}  # from phase2-round2.md; do not change
BACKWARD_WINDOWS = {
    "2000–02 dot-com bust": ("2000-03-24", "2002-10-09"),
    "2008 financial crisis": ("2007-10-09", "2009-03-09"),
}
UNIVERSE = ["SPY", "QQQ", "VGTSX", "VEIEX", "VUSTX", "VFITX", "GC=F", "VGSIX"]
DEFENSIVE = {"VFISX": 0.5, "GC=F": 0.5}
SIXTY = {"SPY": 0.6, "VBMFX": 0.4}


def run_backward(prices: pd.DataFrame, rf: pd.Series, universe: list[str] = UNIVERSE,
                 defensive: dict[str, float] = DEFENSIVE, sixty: dict[str, float] = SIXTY,
                 start: str = "2000-01-01", end: str = "2009-12-31", min_train: int = 750,
                 hmm_refit_every: int = 63, lookback: int = 252, min_history: int = 252,
                 cost_bps: float = 10.0, bootstrap_n: int = 1000) -> dict:
    # Nothing after the test period is used: signals and backtests are cut at `end`.
    prices = prices.loc[:pd.Timestamp(end)]
    prices = prices[prices["SPY"].notna()]
    signal = hmm_stress_proba(prices["SPY"].pct_change(fill_method=None), min_train=min_train,
                              refit_every=hmm_refit_every)
    tilt = percentile_tilt(signal, LOCKED_RULE["q"], min_history=min_history)
    cols = list(dict.fromkeys(universe + list(defensive) + list(sixty)))
    R = simple_returns(prices[[c for c in cols if c in prices.columns]])
    res = evaluate_rule(R, rf, universe, tilt, pd.Timestamp(start), pd.Timestamp(end), BACKWARD_WINDOWS,
                        defensive, sixty, lookback, cost_bps, bootstrap_n, signal=signal)
    return _clean({"rule": LOCKED_RULE, "period": {"start": start, "end": end},
                   "signal_first_date": signal.first_valid_index(), **res})

import json

import numpy as np
import pandas as pd
import pytest

from conftest import regime_prices
from rapo.regime.features import build_features
from rapo.regime.round2 import CANDIDATE_SIGNALS, run_round2

SPLIT = "2014-01-01"


def setup(seed=11, shock_after_split=False):
    px = regime_prices(n=1400, seed=seed)
    rng = np.random.default_rng(3)
    px["EFA"] = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, len(px)))
    if shock_after_split:
        after = px.index >= SPLIT
        noise = np.random.default_rng(99).normal(1, 0.05, (after.sum(), px.shape[1]))
        px.loc[after] = px.loc[after].to_numpy() * noise
    f = build_features(px)
    rng2 = np.random.default_rng(5)
    signals = {"logistic": pd.Series(rng2.uniform(size=len(f)), index=f.index),
               "hmm": pd.Series(rng2.uniform(size=len(f)), index=f.index),
               "vol_21": f["vol_21"]}
    rf = pd.Series(0.0001, index=px.index)
    return px, rf, signals


@pytest.fixture(scope="module")
def result():
    px, rf, signals = setup()
    return run_round2(px, rf, ["SPY", "EFA", "TLT", "GLD"], signals, split=SPLIT, lookback=126,
                      min_history=126, bootstrap_n=200)


def test_six_candidates_scored_on_development(result):
    dev = result["development"]
    assert len(dev) == 6
    assert {(c["signal"], c["q"]) for c in dev} == {(s, q) for s in CANDIDATE_SIGNALS for q in (0.8, 0.9)}
    assert all(c["score"] is not None for c in dev)


def test_chosen_is_the_best_development_score(result):
    best = max(result["development"], key=lambda c: c["score"])
    assert (result["chosen"]["signal"], result["chosen"]["q"]) == (best["signal"], best["q"])


def test_holdout_reports_only_the_chosen_rule(result):
    names = {r["name"] for r in result["holdout"]["table"]}
    assert {"SPY", "60/40", "Equal weight", "Equal weight + regime", "HRP", "HRP + regime"} == names
    assert len(result["holdout"]["bootstrap"]) == 4
    assert result["periods"]["holdout_start"] >= SPLIT


def test_selection_is_blind_to_the_holdout():
    px, rf, signals = setup(shock_after_split=True)
    shocked = run_round2(px, rf, ["SPY", "EFA", "TLT", "GLD"], signals, split=SPLIT, lookback=126,
                         min_history=126, bootstrap_n=50)
    px0, rf0, signals0 = setup()
    base = run_round2(px0, rf0, ["SPY", "EFA", "TLT", "GLD"], signals0, split=SPLIT, lookback=126,
                      min_history=126, bootstrap_n=50)
    assert shocked["chosen"] == base["chosen"]
    assert shocked["development"] == base["development"]


def test_json_safe(result):
    assert "NaN" not in json.dumps(result, allow_nan=False)

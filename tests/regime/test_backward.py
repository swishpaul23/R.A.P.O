import json

import numpy as np
import pandas as pd
import pytest

from conftest import regime_prices
from rapo.regime.backward import BACKWARD_WINDOWS, run_backward

PROXY = {"TLT": "VUSTX", "SHY": "VFISX", "GLD": "GC=F", "AGG": "VBMFX"}
UNIVERSE = ["SPY", "VGTSX", "VUSTX", "GC=F"]
DEFENSIVE = {"VFISX": 0.5, "GC=F": 0.5}
SIXTY = {"SPY": 0.6, "VBMFX": 0.4}


def setup(shock_after=None):
    px = regime_prices(n=1600, seed=21).rename(columns=PROXY)
    px.index = pd.bdate_range("1996-01-01", periods=len(px))
    rng = np.random.default_rng(8)
    px["VGTSX"] = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, len(px)))
    if shock_after:
        after = px.index > shock_after
        px.loc[after] = px.loc[after].to_numpy() * np.random.default_rng(1).normal(1, 0.05, (after.sum(), px.shape[1]))
    rf = pd.Series(0.0001, index=px.index)
    return px, rf


KW = dict(universe=UNIVERSE, defensive=DEFENSIVE, sixty=SIXTY, start="2000-01-01", end="2001-06-29",
          min_train=500, lookback=126, min_history=126, bootstrap_n=100)


@pytest.fixture(scope="module")
def result():
    px, rf = setup()
    return run_backward(px, rf, **KW)


def test_report_shape(result):
    assert {r["name"] for r in result["table"]} == {"SPY", "60/40", "Equal weight", "Equal weight + regime",
                                                    "HRP", "HRP + regime"}
    assert len(result["bootstrap"]) == 4
    assert result["rule"] == {"signal": "hmm", "q": 0.9}
    assert result["series"]["dates"][0] >= "2000-01-01" and result["series"]["dates"][-1] <= "2001-06-29"


def test_dot_com_window_reported_when_covered(result):
    assert "2000–02 dot-com bust" in result["stress_windows"]
    assert set(BACKWARD_WINDOWS) == {"2000–02 dot-com bust", "2008 financial crisis"}


def test_data_after_the_test_period_cannot_change_it():
    px, rf = setup(shock_after="2001-06-29")
    shocked = run_backward(px, rf, **KW)
    px0, rf0 = setup()
    base = run_backward(px0, rf0, **KW)
    assert shocked["table"] == base["table"]


def test_json_safe(result):
    assert "NaN" not in json.dumps(result, allow_nan=False)

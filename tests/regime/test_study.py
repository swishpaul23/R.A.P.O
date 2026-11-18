import json

import numpy as np
import pandas as pd
import pytest

from conftest import regime_prices
from rapo.regime.study import MODELS, run_study


@pytest.fixture(scope="module")
def result():
    px = regime_prices(n=1400, seed=11)
    rng = np.random.default_rng(3)
    px["EFA"] = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, len(px)))
    holiday = px.iloc[[900]].copy()
    holiday.index = [px.index[900] + pd.Timedelta(days=1)]
    holiday.loc[:, [c for c in px.columns if c != "^VIX"]] = np.nan  # VIX quotes, market closed
    px = pd.concat([px, holiday]).sort_index()
    rf = pd.Series(0.0001, index=px.index)
    return run_study(px, rf, universe=["SPY", "EFA", "TLT", "GLD"], start="2010-01-01",
                     min_train=300, refit_every=126, hmm_refit_every=252, lookback=126)


def test_has_every_section(result):
    assert {"as_of", "chosen_model", "forecast", "calibration", "backtests", "stress_windows",
            "bootstrap", "series", "current", "settings"} <= set(result)
    assert set(result["forecast"]) == set(MODELS)
    assert result["chosen_model"] in MODELS


def test_backtest_rows(result):
    names = {r["name"] for r in result["backtests"]}
    assert {"SPY", "60/40", "Equal weight", "HRP", "Equal weight + regime", "HRP + regime"} <= names


def test_json_safe(result):
    text = json.dumps(result, allow_nan=False)
    assert "NaN" not in text


def test_series_lengths_match(result):
    s = result["series"]
    n = len(s["dates"])
    assert n > 0
    for key in ("proba", "tilt", "spy", "hrp", "hrp_regime"):
        assert len(s[key]) == n


def test_bootstrap_entries(result):
    for entry in result["bootstrap"]:
        assert {"base", "metric", "diff", "lo", "hi"} <= set(entry)
        assert entry["lo"] <= entry["diff"] <= entry["hi"] or entry["diff"] is None


def test_current_reading(result):
    c = result["current"]
    assert 0 <= c["proba"] <= 1 and 0 <= c["tilt"] <= 0.4


def test_every_backtest_row_has_a_drawdown(result):
    assert all(r["max_drawdown"] is not None for r in result["backtests"])


def test_forecast_reports_climatology_and_skill(result):
    ref = result["forecast_reference"]
    assert {"brier", "n"} <= set(ref)
    for m in MODELS:
        assert "brier_skill" in result["forecast"][m]

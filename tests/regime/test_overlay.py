import numpy as np
import pandas as pd
import pytest

from rapo.regime.overlay import make_regime_overlay, make_tilt_overlay, percentile_tilt, tilt_from_proba
from rapo.strategies import equal_weight


@pytest.fixture
def window():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=100)
    return pd.DataFrame(rng.normal(0, 0.01, (100, 4)), index=idx, columns=["SPY", "EFA", "SHY", "GLD"])


def const(p, window):
    return pd.Series(p, index=window.index)


def overlay(p, window, **kw):
    return make_regime_overlay(equal_weight, const(p, window), base_assets=["SPY", "EFA"], **kw)(window)


def test_tilt_formula():
    p = pd.Series([0.0, 0.2, 0.6, 1.0])
    assert tilt_from_proba(p, span=1).tolist() == pytest.approx([0.0, 0.0, 0.2, 0.4])


def test_zero_probability_is_the_base_strategy(window):
    assert overlay(0.0, window).to_dict() == pytest.approx({"SPY": 0.5, "EFA": 0.5, "SHY": 0.0, "GLD": 0.0})


def test_full_probability_tilts_cap_into_defensive(window):
    w = overlay(1.0, window)
    assert w.to_dict() == pytest.approx({"SPY": 0.3, "EFA": 0.3, "SHY": 0.2, "GLD": 0.2})


def test_nan_probability_means_no_tilt(window):
    assert overlay(np.nan, window).to_dict() == pytest.approx({"SPY": 0.5, "EFA": 0.5, "SHY": 0.0, "GLD": 0.0})


def test_ineligible_defensive_asset_shifts_weight_to_the_other(window):
    w = make_regime_overlay(equal_weight, const(1.0, window), base_assets=["SPY", "EFA"])(window.drop(columns="SHY"))
    assert w.to_dict() == pytest.approx({"SPY": 0.3, "EFA": 0.3, "GLD": 0.4})


def test_no_defensive_assets_keeps_base(window):
    w = make_regime_overlay(equal_weight, const(1.0, window), base_assets=["SPY", "EFA"])(window[["SPY", "EFA"]])
    assert w.to_dict() == pytest.approx({"SPY": 0.5, "EFA": 0.5})


def test_uses_probability_as_of_the_window_end(window):
    p = const(0.0, window)
    p.loc[window.index[-1] + pd.offsets.BDay(1)] = 1.0  # tomorrow's reading must not leak in
    w = make_regime_overlay(equal_weight, p, base_assets=["SPY", "EFA"])(window)
    assert w["SHY"] == 0.0


def test_weights_valid(window):
    for p in (0.0, 0.33, 0.7, 1.0):
        w = overlay(p, window)
        assert (w >= 0).all() and w.sum() == pytest.approx(1.0)


def test_percentile_tilt_is_zero_without_enough_history():
    s = pd.Series(np.arange(300.0), index=pd.bdate_range("2020-01-01", periods=300))
    t = percentile_tilt(s, q=0.8, min_history=252, span=1)
    assert (t.iloc[:251] == 0).all()


def test_percentile_tilt_is_full_when_signal_hits_new_highs():
    s = pd.Series(np.arange(400.0), index=pd.bdate_range("2020-01-01", periods=400))
    t = percentile_tilt(s, q=0.8, min_history=252, span=1)
    assert t.iloc[-1] == pytest.approx(0.40)


def test_percentile_tilt_is_zero_when_signal_is_low():
    s = pd.Series(np.r_[np.arange(300.0), np.zeros(50)], index=pd.bdate_range("2020-01-01", periods=350))
    t = percentile_tilt(s, q=0.8, min_history=252, span=1)
    assert t.iloc[-1] == 0.0


def test_percentile_tilt_has_no_lookahead():
    rng = np.random.default_rng(4)
    s = pd.Series(rng.normal(size=500), index=pd.bdate_range("2020-01-01", periods=500))
    shocked = s.copy()
    shocked.iloc[401:] = 100.0
    a, b = percentile_tilt(s, q=0.9), percentile_tilt(shocked, q=0.9)
    assert a.iloc[:401].equals(b.iloc[:401])


def test_tilt_overlay_uses_given_tilt(window):
    t = const(0.4, window)
    w = make_tilt_overlay(equal_weight, t, base_assets=["SPY", "EFA"])(window)
    assert w.to_dict() == pytest.approx({"SPY": 0.3, "EFA": 0.3, "SHY": 0.2, "GLD": 0.2})

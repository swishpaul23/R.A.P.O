import numpy as np
import pandas as pd
import pytest

from rapo.regime.overlay import make_regime_overlay, tilt_from_proba
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

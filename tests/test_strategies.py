import pandas as pd
import pytest

from conftest import two_uncorrelated
from rapo.strategies import equal_weight, fixed_weights, inverse_vol, make_max_sharpe, min_variance

ALL = [equal_weight, inverse_vol, min_variance, make_max_sharpe()]
IDS = ["equal_weight", "inverse_vol", "min_variance", "max_sharpe"]


@pytest.mark.parametrize("strat", ALL, ids=IDS)
def test_weights_valid(strat, random_returns):
    w = strat(random_returns)
    assert list(w.index) == list(random_returns.columns)
    assert (w >= -1e-10).all()
    assert w.sum() == pytest.approx(1.0)


def test_equal_weight_values(random_returns):
    assert equal_weight(random_returns).to_numpy() == pytest.approx([0.25] * 4)


def test_inverse_vol_ratio():
    w = inverse_vol(two_uncorrelated())
    assert w["LO"] == pytest.approx(2 / 3, abs=1e-2)
    assert w["HI"] == pytest.approx(1 / 3, abs=1e-2)


def test_min_variance_prefers_low_vol():
    w = min_variance(two_uncorrelated())
    assert w["LO"] > 0.5
    assert w["LO"] == pytest.approx(0.8, abs=0.02)  # inverse-variance for uncorrelated assets


def test_max_sharpe_respects_cap(random_returns):
    assert make_max_sharpe(cap=0.40)(random_returns).max() <= 0.40 + 1e-6


def test_max_sharpe_falls_back_when_all_below_rf(random_returns):
    w = make_max_sharpe(rf_annual=10.0)(random_returns)
    pd.testing.assert_series_equal(w, min_variance(random_returns), atol=1e-6)


@pytest.mark.parametrize("strat", ALL, ids=IDS)
def test_near_duplicate_assets(strat, random_returns):
    r = random_returns.assign(DUP=random_returns.iloc[:, 0] + 1e-9)
    w = strat(r)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-10).all()


def test_fixed_weights_ignores_assets_outside_the_target(random_returns):
    w = fixed_weights({"AAA": 0.6, "BBB": 0.4})(random_returns)
    assert w.to_dict() == pytest.approx({"AAA": 0.6, "BBB": 0.4, "CCC": 0.0, "DDD": 0.0})


def test_fixed_weights_renormalizes_over_eligible(random_returns):
    w = fixed_weights({"AAA": 0.6, "ZZZ": 0.4})(random_returns[["AAA", "BBB"]])
    assert w.to_dict() == pytest.approx({"AAA": 1.0, "BBB": 0.0})


def test_fixed_weights_with_no_target_assets_falls_back_to_equal_weight(random_returns):
    w = fixed_weights({"ZZZ": 1.0})(random_returns)
    assert w.to_numpy() == pytest.approx([0.25] * 4)

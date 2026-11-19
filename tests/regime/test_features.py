import numpy as np
import pandas as pd
import pytest

from rapo.regime.features import FEATURES, build_features


def test_columns(regime_px):
    assert list(build_features(regime_px).columns) == FEATURES


def test_no_lookahead(regime_px):
    base = build_features(regime_px)
    shocked = regime_px.copy()
    shocked.iloc[401:] *= 2
    after = build_features(shocked)
    assert np.allclose(base.iloc[:401].to_numpy(), after.iloc[:401].to_numpy(), equal_nan=True)


def test_missing_inputs_raise(regime_px):
    with pytest.raises(ValueError, match=r"\^VIX"):
        build_features(regime_px.drop(columns="^VIX"))


def test_warmup_only_nan(regime_px):
    f = build_features(regime_px)
    assert not f.iloc[200:].isna().any().any()


def test_curve_is_nan_without_irx(regime_px):
    f = build_features(regime_px.drop(columns="^IRX"))
    assert f["curve"].isna().all()


def test_index_only_days_do_not_poison_later_features(regime_px):
    # ^VIX sometimes quotes on days the stock market is closed (SPY missing).
    holiday = regime_px.index[700] + pd.Timedelta(days=1)
    extra = regime_px.iloc[[700]].copy()
    extra.index = [holiday]
    extra[["SPY", "TLT", "IEF", "HYG", "^TNX", "^IRX"]] = np.nan
    px = pd.concat([regime_px, extra]).sort_index()
    f = build_features(px)
    assert holiday not in f.index
    assert not f.iloc[750:].isna().any().any()
    assert np.allclose(f.to_numpy(), build_features(regime_px).to_numpy(), equal_nan=True)


def test_rate_days_missing_are_carried_forward(regime_px):
    px = regime_px.copy()
    px.iloc[600, px.columns.get_loc("^TNX")] = np.nan
    assert not build_features(px).iloc[600:].isna().any().any()

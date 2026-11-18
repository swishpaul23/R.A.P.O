import numpy as np
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

import numpy as np
import pandas as pd
import pytest

from conftest import two_uncorrelated
from rapo.hrp import hrp, hrp_linkage


def test_hrp_valid(random_returns):
    w = hrp(random_returns)
    assert list(w.index) == list(random_returns.columns)
    assert (w >= 0).all()
    assert w.sum() == pytest.approx(1.0)


def test_hrp_two_uncorrelated_is_inverse_variance():
    w = hrp(two_uncorrelated())
    assert w["LO"] == pytest.approx(0.8, abs=0.02)
    assert w["HI"] == pytest.approx(0.2, abs=0.02)


def test_hrp_single_asset(random_returns):
    w = hrp(random_returns[["AAA"]])
    assert w.iloc[0] == 1.0


def test_hrp_near_duplicates(random_returns):
    r = random_returns.assign(DUP=random_returns.iloc[:, 0] + 1e-9)
    w = hrp(r)
    assert w.sum() == pytest.approx(1.0)
    assert np.isfinite(w).all()


def test_hrp_linkage_shape(random_returns):
    assert hrp_linkage(random_returns).shape == (3, 4)

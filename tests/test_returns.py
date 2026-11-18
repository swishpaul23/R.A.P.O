import numpy as np
import pandas as pd
import pytest

from rapo.returns import log_returns, simple_returns


def test_simple_returns_values():
    p = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=pd.date_range("2024-01-01", periods=3))
    r = simple_returns(p)
    assert r["A"].iloc[0] == pytest.approx(0.10)
    assert r["A"].iloc[1] == pytest.approx(-0.10)
    assert len(r) == 2


def test_simple_returns_preserves_pre_inception_nan():
    p = pd.DataFrame(
        {"OLD": [1.0, 1.1, 1.2, 1.3], "NEW": [np.nan, np.nan, 5.0, 5.5]},
        index=pd.date_range("2024-01-01", periods=4),
    )
    r = simple_returns(p)
    assert len(r) == 3
    assert r["NEW"].isna().sum() == 2
    assert r["NEW"].iloc[-1] == pytest.approx(0.10)


def test_log_returns_match_simple():
    p = pd.DataFrame({"A": [100.0, 110.0]}, index=pd.date_range("2024-01-01", periods=2))
    assert log_returns(p)["A"].iloc[0] == pytest.approx(np.log(1.1))

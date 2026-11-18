import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def random_returns() -> pd.DataFrame:
    """300 days x 4 assets with different drifts and vols (seed 0)."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300)
    mu = np.array([0.0006, 0.0003, 0.0002, 0.0008])
    sigma = np.array([0.010, 0.006, 0.004, 0.020])
    data = rng.normal(mu, sigma, size=(300, 4))
    return pd.DataFrame(data, index=idx, columns=["AAA", "BBB", "CCC", "DDD"])


def two_uncorrelated(n: int = 5000, vols=(0.01, 0.02), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-01", periods=n)
    return pd.DataFrame(rng.normal(0, vols, size=(n, 2)), index=idx, columns=["LO", "HI"])


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Geometric random walks for 3 assets + a benchmark, 2019-2023."""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2019-01-01", "2023-12-29")
    mu = np.array([0.0004, 0.0002, 0.0003, 0.0003])
    sigma = np.array([0.012, 0.005, 0.009, 0.010])
    r = rng.normal(mu, sigma, size=(len(idx), 4))
    prices = 100 * np.cumprod(1 + r, axis=0)
    return pd.DataFrame(prices, index=idx, columns=["EQ", "BOND", "GOLD", "BENCH"])


@pytest.fixture
def flat_rf(synthetic_prices) -> pd.Series:
    return pd.Series(0.0001, index=synthetic_prices.index)

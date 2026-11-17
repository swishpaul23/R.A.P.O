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

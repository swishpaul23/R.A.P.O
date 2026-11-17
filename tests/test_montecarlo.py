import time

import numpy as np
import pandas as pd
import pytest

from rapo.montecarlo import random_portfolios


@pytest.fixture
def inputs():
    tickers = ["A", "B", "C"]
    mu = pd.Series([0.08, 0.05, 0.12], index=tickers)
    vols = np.array([0.15, 0.08, 0.25])
    corr = np.array([[1, 0.2, 0.5], [0.2, 1, 0.1], [0.5, 0.1, 1]])
    cov = pd.DataFrame(np.outer(vols, vols) * corr, index=tickers, columns=tickers)
    return mu, cov


def test_matches_loop_computation(inputs):
    mu, cov = inputs
    df = random_portfolios(mu, cov, n=50, rf=0.02, seed=1)
    W = df[list(mu.index)].to_numpy()
    for i in range(50):
        vol = np.sqrt(W[i] @ cov.to_numpy() @ W[i])
        ret = W[i] @ mu.to_numpy()
        assert df["vol"].iloc[i] == pytest.approx(vol)
        assert df["ret"].iloc[i] == pytest.approx(ret)
        assert df["sharpe"].iloc[i] == pytest.approx((ret - 0.02) / vol)


def test_weights_valid(inputs):
    mu, cov = inputs
    W = random_portfolios(mu, cov, n=1000)[list(mu.index)]
    assert (W >= 0).all().all()
    assert W.sum(axis=1).to_numpy() == pytest.approx(np.ones(1000))


def test_is_fast():
    k = 10
    tickers = [f"T{i}" for i in range(k)]
    mu = pd.Series(np.linspace(0.02, 0.12, k), index=tickers)
    cov = pd.DataFrame(np.eye(k) * 0.04, index=tickers, columns=tickers)
    start = time.perf_counter()
    random_portfolios(mu, cov, n=100_000)
    assert time.perf_counter() - start < 1.0

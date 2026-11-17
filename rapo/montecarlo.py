"""Random-portfolio cloud for the efficient-frontier chart, fully vectorized.

Weights are drawn uniformly from the simplex (Dirichlet with all alphas = 1),
then every portfolio's return and volatility is computed in one matrix
operation instead of a Python loop.
"""

import numpy as np
import pandas as pd


def random_portfolios(mu: pd.Series, cov: pd.DataFrame, n: int = 10_000,
                      rf: float = 0.0, seed: int = 0) -> pd.DataFrame:
    tickers = list(mu.index)
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(len(tickers)), size=n)              # (n, k)
    C = cov.loc[tickers, tickers].to_numpy()
    ret = W @ mu.to_numpy()                                       # (n,)
    vol = np.sqrt(np.einsum("ij,jk,ik->i", W, C, W))              # w_i' C w_i for every row
    out = pd.DataFrame(W, columns=tickers)
    out.insert(0, "sharpe", (ret - rf) / vol)
    out.insert(0, "vol", vol)
    out.insert(0, "ret", ret)
    return out

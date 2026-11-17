"""Expected-return and covariance estimates, annualized (252 trading days)."""

import pandas as pd
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252


def annual_mean(returns: pd.DataFrame) -> pd.Series:
    """Sample mean of daily simple returns x 252."""
    return returns.mean() * TRADING_DAYS


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrunk covariance x 252.

    Shrinkage pulls the noisy sample covariance toward a structured target,
    which keeps optimizers from betting heavily on estimation error and keeps
    the matrix invertible even when assets are nearly identical.
    """
    lw = LedoitWolf().fit(returns.to_numpy())
    return pd.DataFrame(lw.covariance_ * TRADING_DAYS, index=returns.columns, columns=returns.columns)

"""Price -> return conversions.

Missing prices before an asset's first trading day stay NaN. We never drop a
whole row because one asset is missing; that would silently shorten every
other asset's history.
"""

import numpy as np
import pandas as pd


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns P_t / P_{t-1} - 1. Drops only the first row."""
    return prices.pct_change(fill_method=None).iloc[1:]


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns ln(P_t / P_{t-1}). Drops only the first row."""
    return np.log(prices / prices.shift(1)).iloc[1:]

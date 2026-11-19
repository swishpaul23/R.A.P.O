"""Stress labels: was the next month a market-stress period?"""

import numpy as np
import pandas as pd


def stress_label(spy: pd.Series, horizon: int = 21, threshold: float = -0.08) -> pd.Series:
    """1.0 if SPY's lowest close in the next `horizon` days is <= `threshold` below today's close.

    The last `horizon` rows have no complete future window and are NaN. These labels
    look forward by design: models must only train on labels whose window has ended.
    """
    future_min = spy[::-1].rolling(horizon).min()[::-1].shift(-1)
    worst = future_min / spy - 1.0
    lab = (worst <= threshold + 1e-12).astype(float)
    lab[worst.isna()] = np.nan
    return lab.rename("stress")

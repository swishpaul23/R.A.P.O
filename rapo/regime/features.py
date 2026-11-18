"""Daily features describing the market regime.

Every value dated d uses only prices dated <= d (rolling windows, no centring,
no negative shifts). Inputs are price levels; ^VIX, ^TNX and ^IRX are index
levels (percent), not prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["SPY", "TLT", "IEF", "HYG", "^VIX", "^TNX"]
FEATURES = ["vol_21", "vol_63", "vol_ratio", "vix", "vix_chg_21", "corr_eq_bond_63",
            "credit_63", "rate_mom_63", "curve", "mom_200"]


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    missing = [t for t in REQUIRED if t not in prices.columns]
    if missing:
        raise ValueError(f"Regime features need price data for: {', '.join(missing)}. "
                         "Re-run scripts/make_snapshot.py.")
    # Work on stock-market trading days only: index levels (^VIX) sometimes quote on
    # exchange holidays, and a missing SPY day would otherwise blank out every rolling window.
    p = prices.sort_index()
    p = p[p["SPY"].notna()].copy()
    levels = [c for c in p.columns if c.startswith("^")]
    p[levels] = p[levels].ffill()  # rate/vol index missing on a trading day: carry the last value
    spy_r = p["SPY"].pct_change(fill_method=None)
    tlt_r = p["TLT"].pct_change(fill_method=None)

    f = pd.DataFrame(index=p.index)
    f["vol_21"] = spy_r.rolling(21).std() * np.sqrt(252)
    f["vol_63"] = spy_r.rolling(63).std() * np.sqrt(252)
    f["vol_ratio"] = f["vol_21"] / f["vol_63"]
    f["vix"] = p["^VIX"]
    f["vix_chg_21"] = p["^VIX"] - p["^VIX"].shift(21)
    f["corr_eq_bond_63"] = spy_r.rolling(63).corr(tlt_r)
    f["credit_63"] = p["HYG"].pct_change(63, fill_method=None) - p["IEF"].pct_change(63, fill_method=None)
    f["rate_mom_63"] = p["^TNX"] - p["^TNX"].shift(63)
    f["curve"] = p["^TNX"] - p["^IRX"] if "^IRX" in p.columns else np.nan
    f["mom_200"] = p["SPY"] / p["SPY"].rolling(200).mean() - 1.0
    return f[FEATURES]
